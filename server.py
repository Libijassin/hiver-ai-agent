"""
Server: Web API and static server for SpotifyCares AI Support Agent Web App.
Runs on http://localhost:8000
"""
import sys
import json
import os
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Setup paths
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
WEB_DIR = PROJECT_ROOT / "web"

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from config import RESULTS_DIR, GOLDEN_DIR, PROCESSED_DIR
from src.retriever import HistoricalRetriever, load_reference_pairs
from src.agent import SupportAgent

# Global singletons
agent_instance = None
cached_metrics = {}
cached_golden = []
cached_failures = {}
cached_audit = {}
cached_decisions = {}
cached_human_agreement = {}

def init_agent():
    global agent_instance, cached_metrics, cached_golden, cached_failures, cached_audit, cached_decisions, cached_human_agreement
    print("[Server] Initializing SpotifyCares Support Agent...")

    # Load RAG retriever
    pairs = load_reference_pairs()
    retriever = HistoricalRetriever(pairs)
    agent_instance = SupportAgent(retriever)
    print(f"[Server] SupportAgent initialized (Mode: {'DEMO/MOCK' if agent_instance.use_mock else 'REAL LLM (Gemini)'})")

    # Load metrics cache
    try:
        baseline_cmp = {}
        if (RESULTS_DIR / "baseline_comparison.json").exists():
            with open(RESULTS_DIR / "baseline_comparison.json", "r", encoding="utf-8") as f:
                baseline_cmp = json.load(f)

        agent_eval = {}
        if (RESULTS_DIR / "evaluation_agent.json").exists():
            with open(RESULTS_DIR / "evaluation_agent.json", "r", encoding="utf-8") as f:
                agent_eval = json.load(f)

        judge_scores = {}
        if (RESULTS_DIR / "judge_scores.json").exists():
            with open(RESULTS_DIR / "judge_scores.json", "r", encoding="utf-8") as f:
                judge_scores = json.load(f)

        cached_metrics = {
            "baseline_comparison": baseline_cmp,
            "agent_evaluation": agent_eval,
            "judge_scores": judge_scores,
        }
    except Exception as e:
        print(f"[Server] Warning loading metrics: {e}")

    # Load golden set cache
    try:
        if (RESULTS_DIR / "predictions_agent.json").exists():
            with open(RESULTS_DIR / "predictions_agent.json", "r", encoding="utf-8") as f:
                cached_golden = json.load(f)
        elif (GOLDEN_DIR / "golden_eval_set.json").exists():
            with open(GOLDEN_DIR / "golden_eval_set.json", "r", encoding="utf-8") as f:
                golden_raw = json.load(f)
                cached_golden = golden_raw
    except Exception as e:
        print(f"[Server] Warning loading golden predictions: {e}")

    # Load failure analysis cache
    try:
        if (RESULTS_DIR / "failure_analysis_agent.json").exists():
            with open(RESULTS_DIR / "failure_analysis_agent.json", "r", encoding="utf-8") as f:
                cached_failures = json.load(f)
    except Exception as e:
        print(f"[Server] Warning loading failure analysis: {e}")

    # Load headline audit cache
    try:
        if (RESULTS_DIR / "headline_audit.json").exists():
            with open(RESULTS_DIR / "headline_audit.json", "r", encoding="utf-8") as f:
                cached_audit = json.load(f)
    except Exception as e:
        print(f"[Server] Warning loading headline audit: {e}")

    # Load decision log cache
    try:
        if (RESULTS_DIR / "decision_log.json").exists():
            with open(RESULTS_DIR / "decision_log.json", "r", encoding="utf-8") as f:
                cached_decisions = json.load(f)
    except Exception as e:
        print(f"[Server] Warning loading decision log: {e}")

    # Load human agreement cache
    try:
        if (RESULTS_DIR / "human_agreement.json").exists():
            with open(RESULTS_DIR / "human_agreement.json", "r", encoding="utf-8") as f:
                cached_human_agreement = json.load(f)
    except Exception as e:
        print(f"[Server] Warning loading human agreement: {e}")


class AgentRequestHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Route static request to web/ directory
        parsed_path = urllib.parse.urlparse(path).path
        if parsed_path == "/" or parsed_path == "":
            parsed_path = "/index.html"
        
        # Clean relative path
        rel_path = parsed_path.lstrip("/")
        full_path = WEB_DIR / rel_path
        return str(full_path)

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path).path
        
        if parsed_path == "/api/metrics":
            self.send_json_response(cached_metrics)
        elif parsed_path == "/api/golden_set":
            self.send_json_response(cached_golden)
        elif parsed_path == "/api/failure_analysis":
            self.send_json_response(cached_failures)
        elif parsed_path == "/api/headline_audit":
            self.send_json_response(cached_audit)
        elif parsed_path == "/api/decision_log":
            self.send_json_response(cached_decisions)
        elif parsed_path == "/api/human_agreement":
            self.send_json_response(cached_human_agreement)
        elif parsed_path == "/api/status":
            self.send_json_response({
                "status": "online",
                "mode": "DEMO/MOCK" if agent_instance and agent_instance.use_mock else "REAL LLM (Gemini)",
                "reference_pairs": len(agent_instance.retriever.reference_pairs) if agent_instance else 0
            })
        else:
            # Fallback to static file server
            super().do_GET()


    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path).path
        
        if parsed_path == "/api/process_tweet":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                tweet = data.get("tweet") or data.get("customer_message") or ""
                
                if not tweet.strip():
                    self.send_json_response({"error": "Tweet content cannot be empty"}, status=400)
                    return
                
                if agent_instance is None:
                    self.send_json_response({"error": "Agent not initialized"}, status=500)
                    return
                
                result = agent_instance.process(tweet)
                self.send_json_response(result)
            except Exception as e:
                self.send_json_response({"error": str(e)}, status=500)
        else:
            self.send_error(404, "Endpoint not found")

    def send_json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server(port=8000):
    init_agent()
    
    server_address = ("127.0.0.1", port)
    try:
        httpd = HTTPServer(server_address, AgentRequestHandler)
        print(f"\n============================================================")
        print(f"🚀 SpotifyCares AI Agent Web App is running!")
        print(f"   URL: http://localhost:{port}")
        print(f"============================================================\n")
        httpd.serve_forever()
    except OSError as e:
        if port < 8010:
            print(f"Port {port} in use, trying {port + 1}...")
            run_server(port + 1)
        else:
            raise e


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    args = parser.parse_args()
    run_server(args.port)
