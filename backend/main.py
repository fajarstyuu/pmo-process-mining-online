from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError
import logging
import pm4py
import pandas as pd
import io
import os
import tempfile
from typing import Optional
import json

app = FastAPI(title="Process Mining API with Inductive Miner", version="1.0.0")

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS for local development (adjust origins as needed)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware to log incoming requests for /discover-model to help debug 400s
@app.middleware("http")
async def log_discover_model_request(request: Request, call_next):
    try:
        if request.url.path == "/discover-model":
            content_type = request.headers.get("content-type")
            body = await request.body()
            # log a short preview (avoid huge binary dumps)
            try:
                preview = body.decode('utf-8')[:2000]
            except Exception:
                preview = str(body[:2000])

            logger.info(f"[debug] Incoming {request.method} {request.url.path} Content-Type: {content_type} Body-preview: {preview}")
        response = await call_next(request)
        return response
    except Exception as e:
        logger.exception("Unhandled error in middleware while processing request")
        raise


# Exception handlers to return and log validation errors/details
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Request validation error for {request.url.path}: {exc}")
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "body": exc.body})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception for {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Log and return the HTTPException detail so the frontend sees why a 4xx was returned
    try:
        logger.warning(f"HTTPException for {request.url.path}: status_code={exc.status_code} detail={exc.detail}")
    except Exception:
        logger.warning(f"HTTPException for {request.url.path}: {exc}")
    content = {"detail": exc.detail}
    return JSONResponse(status_code=exc.status_code, content=content)

# Global variable to store the current event log
current_event_log = None
current_petri_net = None
current_initial_marking = None
current_final_marking = None

@app.get("/")
def read_root():
    return {
        "message": "Process Mining API with Inductive Miner",
        "endpoints": {
            "upload_log": "/upload-log",
            "discover_model": "/discover-model", 
            "get_model_info": "/model-info",
            "export_petri_net": "/export-petri-net",
            "conformance_check": "/conformance-check",
            "get_log_statistics": "/log-statistics"
        }
    }

@app.post("/upload-log")
async def upload_event_log(file: UploadFile = File(...)):
    """
    Upload an event log file (CSV or XES format)
    """
    global current_event_log
    
    if not file.filename:
        print(HTTPException(status_code=400, detail="No file provided"))
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_extension = file.filename.lower().split('.')[-1]
    
    try:
        # Read file content
        content = await file.read()
        
        if file_extension == 'csv':
            # Parse CSV file
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            
            # Convert to event log format expected by pm4py
            # Assuming columns: case:concept:name, concept:name, time:timestamp
            # You might need to adjust column names based on your CSV structure
            required_columns = ['case:concept:name', 'concept:name', 'time:timestamp']
            
            # Check if required columns exist, if not try common alternatives
            column_mapping = {}
            for req_col in required_columns:
                if req_col not in df.columns:
                    # Try common alternative names
                    alternatives = {
                        'case:concept:name': ['case_id', 'caseid', 'case', 'trace_id'],
                        'concept:name': ['activity', 'event', 'activity_name', 'task'],
                        'time:timestamp': ['timestamp', 'time', 'datetime', 'date']
                    }
                    
                    found = False
                    for alt in alternatives.get(req_col, []):
                        if alt in df.columns:
                            column_mapping[alt] = req_col
                            found = True
                            break
                    
                    if not found:
                        available_cols = list(df.columns)
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Required column '{req_col}' not found. Available columns: {available_cols}"
                        )
            
            # Rename columns if mapping exists
            if column_mapping:
                df = df.rename(columns=column_mapping)
            
            # Convert to event log
            current_event_log = pm4py.format_dataframe(df, 
                                                      case_id='case:concept:name',
                                                      activity_key='concept:name', 
                                                      timestamp_key='time:timestamp')
        
        elif file_extension == 'xes':
            # Save XES file temporarily and load it
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xes') as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                current_event_log = pm4py.read_xes(tmp_file_path)
            finally:
                os.unlink(tmp_file_path)
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please use CSV or XES.")
        
        # Get basic statistics
        num_cases = len(current_event_log['case:concept:name'].unique())
        num_events = len(current_event_log)
        activities = current_event_log['concept:name'].unique().tolist()
        
        return {
            "message": "Event log uploaded successfully",
            "statistics": {
                "number_of_cases": num_cases,
                "number_of_events": num_events,
                "number_of_activities": len(activities),
                "activities": activities
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/discover-model")
async def discover_process_model(file: UploadFile = File(...), noise_threshold: float = Form(0.0), model: str = Form("inductive")):
    """
    Upload an event log (CSV or XES) and discover a Petri net using the Inductive Miner.

    Returns JSON formatted for Cytoscape.js with `nodes` and `edges` arrays plus model statistics.
    """
    global current_event_log, current_petri_net, current_initial_marking, current_final_marking

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Determine extension
    file_extension = file.filename.lower().split('.')[-1]

    try:
        content = await file.read()

        # Parse uploaded file into a DataFrame / event log compatible with pm4py
        if file_extension == 'csv':
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))

            # Robust column detection: normalize column names (lowercase, remove non-alnum)
            import re

            def _norm(s: str) -> str:
                return re.sub(r'[^a-z0-9]', '', str(s).lower())

            cols = list(df.columns)
            norm_map = {c: _norm(c) for c in cols}

            # candidate normalized tokens for required pm4py columns
            candidates = {
                'case:concept:name': ['caseconceptname', 'caseid', 'case', 'traceid', 'case_id', 'case id'],
                'concept:name': ['conceptname', 'activity', 'event', 'activityname', 'task', 'action'],
                'time:timestamp': ['timetimestamp', 'timestamp', 'time', 'datetime', 'date', 'starttimestamp', 'completetimestamp', 'start', 'complete']
            }

            column_mapping = {}

            # Try to find best matches for required columns
            for req_col, cand_list in candidates.items():
                found_col = None
                # exact normalized match first
                for orig_col, n in norm_map.items():
                    if n in cand_list or any(c == n for c in cand_list):
                        found_col = orig_col
                        break

                # fuzzy contains match (e.g., 'caseid' matches 'caseid', 'caseidnumber')
                if not found_col:
                    for orig_col, n in norm_map.items():
                        for cand in cand_list:
                            if cand in n or n in cand:
                                found_col = orig_col
                                break
                        if found_col:
                            break

                if not found_col:
                    # as a last resort try keyword containment for short words
                    for orig_col, n in norm_map.items():
                        for keyword in ['case', 'activity', 'time', 'date', 'timestamp']:
                            if keyword in cand_list[0] and keyword in n:
                                found_col = orig_col
                                break
                        if found_col:
                            break

                if not found_col:
                    available_cols = cols
                    raise HTTPException(
                        status_code=400,
                        detail=(f"Required column '{req_col}' not found. Available columns: {available_cols}")
                    )

                # map the found original column name to the pm4py-required name
                column_mapping[found_col] = req_col

            # If timestamp candidate matched multiple columns (e.g., start and complete), prefer complete or start
            # The mapping logic above already selects the first reasonable match; keep it.

            if column_mapping:
                # rename df columns to pm4py expected names
                df = df.rename(columns=column_mapping)

            event_log = pm4py.format_dataframe(df,
                                               case_id='case:concept:name',
                                               activity_key='concept:name',
                                               timestamp_key='time:timestamp')

        elif file_extension == 'xes':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xes') as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            try:
                event_log = pm4py.read_xes(tmp_file_path)
            finally:
                try:
                    os.unlink(tmp_file_path)
                except Exception:
                    pass
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please use CSV or XES.")

        # Store the uploaded log globally (so other endpoints can use it if needed)
        current_event_log = event_log

        # Discover Petri net using Inductive Miner
        if (model.lower() == "inductive" or model == "" or model is None):
            current_petri_net, current_initial_marking, current_final_marking = pm4py.discover_petri_net_inductive(
                current_event_log,
                noise_threshold=noise_threshold
            )
        elif (model.lower() == "alpha"):
            current_petri_net, current_initial_marking, current_final_marking = pm4py.discover_petri_net_alpha(
                current_event_log
            )
        elif (model.lower() == "heuristic"):
            current_petri_net, current_initial_marking, current_final_marking = pm4py.discover_petri_net_heuristic(
                current_event_log
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type. Use 'inductive', 'alpha', or 'heuristic'.")
        
        # Build Cytoscape.js nodes and edges
        nodes = []
        edges = []

        # Create id maps to keep node ids unique and predictable for Cytoscape
        place_id_map = {p.name: f"p_{p.name}" for p in current_petri_net.places}
        trans_id_map = {t.name: f"t_{t.name}" for t in current_petri_net.transitions}

        # Add place nodes
        for p in current_petri_net.places:
            nodes.append({
                "data": {
                    "id": place_id_map[p.name],
                    "label": p.name,
                    "type": "place"
                }
            })

        # Add transition nodes (visible or invisible)
        for t in current_petri_net.transitions:
            label = t.label if hasattr(t, 'label') and t.label is not None else t.name
            visible = True if (hasattr(t, 'label') and t.label is not None) else False
            nodes.append({
                "data": {
                    "id": trans_id_map[t.name],
                    "label": label,
                    "type": "transition",
                    "visible": visible
                }
            })

        # Add edges from arcs
        for idx, arc in enumerate(current_petri_net.arcs):
            src_name = arc.source.name
            tgt_name = arc.target.name

            if src_name in place_id_map:
                src_id = place_id_map[src_name]
            else:
                src_id = trans_id_map.get(src_name, f"unknown_{src_name}")

            if tgt_name in place_id_map:
                tgt_id = place_id_map[tgt_name]
            else:
                tgt_id = trans_id_map.get(tgt_name, f"unknown_{tgt_name}")

            edges.append({
                "data": {
                    "id": f"e_{idx}",
                    "source": src_id,
                    "target": tgt_id
                }
            })

        # Model statistics
        model_stats = {
            "number_of_places": len(current_petri_net.places),
            "number_of_transitions": len(current_petri_net.transitions),
            "number_of_arcs": len(current_petri_net.arcs),
            "noise_threshold_used": noise_threshold
        }

        response_payload = {
            "message": "Model discovered successfully",
            "cytoscape": {
                "nodes": nodes,
                "edges": edges
            },
            "model_statistics": model_stats
        }

        return JSONResponse(content=response_payload)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error discovering process model: {str(e)}")

@app.get("/model-info")
def get_model_info():
    """
    Get information about the currently discovered model
    """
    global current_petri_net, current_initial_marking, current_final_marking
    
    if current_petri_net is None:
        raise HTTPException(status_code=400, detail="No process model available. Please discover a model first.")
    
    try:
        # Get detailed model information
        places = [{"id": p.name, "name": str(p)} for p in current_petri_net.places]
        
        transitions = []
        for t in current_petri_net.transitions:
            transitions.append({
                "id": t.name,
                "label": t.label,
                "visible": t.label is not None
            })
        
        arcs = []
        for arc in current_petri_net.arcs:
            arcs.append({
                "source": arc.source.name,
                "target": arc.target.name,
                "weight": arc.weight if hasattr(arc, 'weight') else 1
            })
        
        # Initial marking
        initial_marking = {p.name: current_initial_marking[p] for p in current_initial_marking}
        
        # Final marking
        final_marking = {p.name: current_final_marking[p] for p in current_final_marking}
        
        return {
            "model_info": {
                "places": places,
                "transitions": transitions,
                "arcs": arcs,
                "initial_marking": initial_marking,
                "final_marking": final_marking
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.get("/export-petri-net")
def export_petri_net(format: str = "pnml"):
    """
    Export the discovered Petri net in various formats (pnml, dot, etc.)
    """
    global current_petri_net, current_initial_marking, current_final_marking
    
    if current_petri_net is None:
        raise HTTPException(status_code=400, detail="No process model available. Please discover a model first.")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}') as tmp_file:
            tmp_file_path = tmp_file.name
        
        if format.lower() == "pnml":
            pm4py.write_pnml(current_petri_net, current_initial_marking, current_final_marking, tmp_file_path)
            media_type = "application/xml"
        elif format.lower() == "dot":
            pm4py.save_vis_petri_net(current_petri_net, current_initial_marking, current_final_marking, tmp_file_path)
            media_type = "text/plain"
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'pnml' or 'dot'.")
        
        return FileResponse(
            path=tmp_file_path,
            filename=f"petri_net.{format}",
            media_type=media_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting Petri net: {str(e)}")

@app.post("/conformance-check")
async def conformance_check():
    """
    Perform conformance checking between the event log and discovered model
    """
    global current_event_log, current_petri_net, current_initial_marking, current_final_marking
    
    if current_event_log is None or current_petri_net is None:
        raise HTTPException(
            status_code=400, 
            detail="Both event log and process model are required. Please upload log and discover model first."
        )
    
    try:
        # Perform token-based replay for conformance checking
        replayed_traces = pm4py.conformance_diagnostics_token_based_replay(
            current_event_log,
            current_petri_net,
            current_initial_marking,
            current_final_marking
        )
        
        # Calculate fitness statistics
        total_traces = len(replayed_traces)
        perfectly_fitting = sum(1 for trace in replayed_traces if trace['trace_is_fit'])
        
        fitness_scores = [trace['trace_fitness'] for trace in replayed_traces]
        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
        
        # Get detailed conformance results
        conformance_results = []
        for i, trace in enumerate(replayed_traces):
            conformance_results.append({
                "trace_index": i,
                "case_id": trace.get('case_id', f"case_{i}"),
                "is_fit": trace['trace_is_fit'],
                "fitness_score": trace['trace_fitness'],
                "missing_tokens": trace.get('missing_tokens', 0),
                "consumed_tokens": trace.get('consumed_tokens', 0),
                "remaining_tokens": trace.get('remaining_tokens', 0)
            })
        
        return {
            "conformance_statistics": {
                "total_traces": total_traces,
                "perfectly_fitting_traces": perfectly_fitting,
                "fitness_percentage": (perfectly_fitting / total_traces * 100) if total_traces > 0 else 0,
                "average_fitness_score": avg_fitness
            },
            "detailed_results": conformance_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing conformance check: {str(e)}")

@app.get("/log-statistics")
def get_log_statistics():
    """
    Get detailed statistics about the uploaded event log
    """
    global current_event_log
    
    if current_event_log is None:
        raise HTTPException(status_code=400, detail="No event log uploaded. Please upload an event log first.")
    
    try:
        # Basic statistics
        num_cases = len(current_event_log['case:concept:name'].unique())
        num_events = len(current_event_log)
        activities = current_event_log['concept:name'].unique().tolist()
        
        # Case length statistics
        case_lengths = current_event_log.groupby('case:concept:name').size()
        
        # Activity frequency
        activity_freq = current_event_log['concept:name'].value_counts().to_dict()
        
        # Time span (if timestamp is available)
        time_stats = {}
        if 'time:timestamp' in current_event_log.columns:
            current_event_log['time:timestamp'] = pd.to_datetime(current_event_log['time:timestamp'])
            time_stats = {
                "start_time": current_event_log['time:timestamp'].min().isoformat(),
                "end_time": current_event_log['time:timestamp'].max().isoformat(),
                "duration_days": (current_event_log['time:timestamp'].max() - 
                                current_event_log['time:timestamp'].min()).days
            }
        
        # Variants (unique process traces)
        variants = pm4py.get_variants_as_tuples(current_event_log)
        
        return {
            "basic_statistics": {
                "number_of_cases": num_cases,
                "number_of_events": num_events,
                "number_of_activities": len(activities),
                "activities": activities
            },
            "case_statistics": {
                "min_case_length": int(case_lengths.min()),
                "max_case_length": int(case_lengths.max()),
                "avg_case_length": float(case_lengths.mean()),
                "median_case_length": float(case_lengths.median())
            },
            "activity_frequency": activity_freq,
            "time_statistics": time_stats,
            "process_variants": {
                "number_of_variants": len(variants),
                "most_frequent_variants": [
                    {"variant": list(variant), "frequency": freq} 
                    for variant, freq in sorted(variants.items(), key=lambda x: x[1], reverse=True)[:5]
                ]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting log statistics: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "pm4py_version": pm4py.__version__}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
