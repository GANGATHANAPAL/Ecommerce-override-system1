from __future__ import annotations

import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from app.inference.pipeline import analyze_customer, find_demo_customer, load_demo_customers

HOST = "0.0.0.0"
PORT = 8000

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Adaptive Hybrid Customer Retention Demo</title>
    <style>
        :root {
            --bg: #f5f7fb;
            --panel: #ffffff;
            --panel-alt: #eef4ff;
            --primary: #123a7a;
            --muted: #5d6b82;
            --success: #1b9c68;
            --warning: #c77d00;
            --danger: #b33a3a;
            --border: #dfe7f4;
            --shadow: 0 12px 30px rgba(10, 28, 58, 0.08);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: var(--bg);
            color: #182333;
        }
        .container {
            max-width: 1340px;
            margin: 24px auto;
            padding: 0 20px 40px;
        }
        .header {
            background: linear-gradient(135deg, #0f2d5d, #184c89);
            color: white;
            border-radius: 18px;
            padding: 22px 26px;
            box-shadow: var(--shadow);
            margin-bottom: 22px;
        }
        .header h1 { margin: 0 0 8px; font-size: 2.2rem; }
        .header p { margin: 0; opacity: 0.88; }
        .layout {
            display: grid;
            grid-template-columns: 420px 1fr;
            gap: 22px;
        }
        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: var(--shadow);
            padding: 18px 20px;
        }
        .panel h2 {
            margin: 0 0 14px;
            font-size: 1.05rem;
            color: var(--primary);
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        label {
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-bottom: 12px;
            font-size: 0.82rem;
            color: var(--muted);
            font-weight: 600;
        }
        input, select, textarea {
            width: 100%;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: #fff;
            padding: 10px 12px;
            font-size: 0.96rem;
            color: #1f2d3d;
        }
        textarea {
            min-height: 88px;
            resize: vertical;
        }
        .demo-grid, .input-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px 12px;
        }
        .button-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
        }
        button {
            border: none;
            border-radius: 10px;
            padding: 10px 14px;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.1s ease;
        }
        button:hover { transform: translateY(-1px); }
        .primary { background: var(--primary); color: white; }
        .secondary { background: #edf4ff; color: var(--primary); }
        .ghost { background: #f3f6ff; color: #5d6b82; }
        .danger { background: #fce9e9; color: var(--danger); }
        .status-box {
            background: #f7f9fc;
            border: 1px solid var(--border);
            border-left: 4px solid var(--primary);
            border-radius: 10px;
            padding: 10px 14px;
            margin-top: 16px;
            color: var(--muted);
            font-size: 0.9rem;
        }
        .risk-cards, .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-top: 14px;
        }
        .metric-card {
            background: var(--panel-alt);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 15px;
        }
        .metric-card .label {
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-card .value {
            font-size: 1.65rem;
            font-weight: 800;
            margin-top: 8px;
            color: var(--primary);
        }
        .metric-card .sub {
            color: var(--muted);
            font-size: 0.75rem;
            margin-top: 4px;
        }
        .results-stack {
            display: grid;
            gap: 18px;
            margin-top: 6px;
        }
        .section-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
        }
        .badge {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: #eaf7f1;
            color: var(--success);
            font-weight: 800;
            font-size: 0.75rem;
            letter-spacing: 0.04em;
        }
        .badge.warn { background: #fff3db; color: var(--warning); }
        .badge.danger { background: #fbe3e3; color: var(--danger); }
        .pill-row { display: flex; gap: 8px; flex-wrap: wrap; }
        .progress {
            height: 10px;
            background: #edf2f7;
            border-radius: 999px;
            overflow: hidden;
            margin: 10px 0 14px;
        }
        .progress-bar {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #123a7a, #3da0ff);
            transition: width 0.35s ease;
        }
        .ordered-list {
            margin: 0;
            padding-left: 18px;
            color: var(--muted);
            line-height: 1.6;
        }
        .embeddings {
            max-height: 220px;
            overflow: auto;
            background: #f8fbff;
            border-radius: 10px;
            border: 1px solid var(--border);
            padding: 10px;
            font-family: Consolas, monospace;
            font-size: 0.72rem;
        }
        @media (max-width: 980px) {
            .layout { grid-template-columns: 1fr; }
            .section-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Adaptive Hybrid Customer Retention</h1>
            <p>Hybrid churn prediction, behaviour drift, CRPI prioritisation, and recommendation decision pipeline.</p>
        </div>

        <div class="layout">
            <div class="panel">
                <h2>Customer Input</h2>
                <div class="button-row" id="demo-buttons">
                    <button class="secondary" data-demo="DEMO_01">Load Demo Customer 1</button>
                    <button class="secondary" data-demo="DEMO_02">Load Demo Customer 2</button>
                    <button class="secondary" data-demo="DEMO_03">Load Demo Customer 3</button>
                    <button class="secondary" data-demo="DEMO_04">Load Demo Customer 4</button>
                    <button class="secondary" data-demo="DEMO_05">Load Demo Customer 5</button>
                </div>

                <form id="customer-form">
                    <div class="input-grid">
                        <label>Customer ID <input name="customer_id" id="customer_id" value="DEMO_02" /></label>
                        <label>Display Name <input name="display_name" id="display_name" value="Customer Demo 02" /></label>
                        <label>Tenure (months) <input name="tenure_months" id="tenure_months" type="number" step="1" value="39" /></label>
                        <label>Risk Profile <select name="risk_profile" id="risk_profile"><option>Low</option><option selected>Medium</option><option>High</option></select></label>
                        <label>Previous Orders <input name="previous_orders" id="previous_orders" type="number" min="1" step="1" value="20" /></label>
                        <label>Average Order Value <input name="avg_order_value" id="avg_order_value" type="number" step="0.01" value="198.60" /></label>
                        <label>Total Spending <input name="total_spending" id="total_spending" type="number" step="0.01" value="3972.00" /></label>
                        <label>Recency (days) <input name="recency_days" id="recency_days" type="number" step="1" value="52" /></label>
                        <label>Average Days Between Purchases <input name="avg_days_between_purchases" id="avg_days_between_purchases" type="number" step="1" value="48" /></label>
                        <label>Last Order Amount <input name="last_order_amount" id="last_order_amount" type="number" step="0.01" value="166.00" /></label>
                        <label>Average Review Score <input name="avg_review_score" id="avg_review_score" type="number" step="0.1" value="3.6" /></label>
                        <label>Cancelled Orders <input name="cancelled_orders" id="cancelled_orders" type="number" step="1" value="5" /></label>
                        <label>Delivered Orders <input name="delivered_orders" id="delivered_orders" type="number" step="1" value="15" /></label>
                        <label>Payment Attempts <input name="payment_attempts" id="payment_attempts" type="number" step="1" value="23" /></label>
                        <label>Preferred Payment Type <select name="preferred_payment_type" id="preferred_payment_type"><option>Credit Card</option><option selected>Bank Transfer</option><option>Debit Card</option><option>Cash on Delivery</option></select></label>
                        <label>Product Count <input name="product_count" id="product_count" type="number" step="1" value="14" /></label>
                        <label>Seller Count <input name="seller_count" id="seller_count" type="number" step="1" value="10" /></label>
                        <label>Customer Category <input name="category" id="category" value="Electronics" /></label>
                        <label>Recent Purchase Behaviour <textarea name="recent_purchase_behavior" id="recent_purchase_behavior">Recent orders have dropped sharply and the customer now buys less frequently with increasing gaps between orders.</textarea></label>
                        <label>Previous Purchase Behaviour <textarea name="previous_purchase_behavior" id="previous_purchase_behavior">Customer previously purchased regularly but has shown a significant recent regime change in purchase cadence and value.</textarea></label>
                    </div>

                    <div class="button-row">
                        <button type="button" class="primary" id="analyze-btn">Analyze Customer</button>
                        <button type="button" class="ghost" id="reset-btn">Reset</button>
                    </div>
                </form>

                <div class="status-box" id="pipeline-status">Status: Ready for analysis.</div>
            </div>

            <div class="results-stack">
                <div class="panel">
                    <h2>Customer Retention Analysis</h2>
                    <div class="summary-cards" id="summary-cards">
                        <div class="metric-card"><div class="label">Customer</div><div class="value" id="customer-name">Customer Demo 02</div></div>
                        <div class="metric-card"><div class="label">Churn Probability</div><div class="value" id="churn-probability">--</div></div>
                        <div class="metric-card"><div class="label">Customer Value</div><div class="value" id="customer-value">--</div></div>
                        <div class="metric-card"><div class="label">Drift Severity</div><div class="value" id="drift-severity">--</div></div>
                        <div class="metric-card"><div class="label">CRPI</div><div class="value" id="crpi-score">--</div></div>
                    </div>
                </div>

                <div class="panel">
                    <h2>Model Analysis</h2>
                    <div class="section-grid">
                        <div class="metric-card"><div class="label">RF Churn Probability</div><div class="value" id="rf-probability">--</div></div>
                        <div class="metric-card"><div class="label">XGBoost Churn Probability</div><div class="value" id="xgb-probability">--</div></div>
                    </div>
                    <div class="metric-card" style="margin-top: 14px;">
                        <div class="label">GRU Embedding</div>
                        <div class="value" id="gru-embedding-size" style="font-size: 1.2rem;">--</div>
                        <div class="sub">64-D behaviour representation</div>
                        <div class="pill-row" style="margin-top: 10px;">
                            <span class="badge">Fusion: 66 features</span>
                            <span class="badge warn">2 model probabilities + 64 embedding values</span>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <h2>Behaviour Drift</h2>
                    <div class="section-grid">
                        <div class="metric-card"><div class="label">Drift Score</div><div class="value" id="drift-score">--</div></div>
                        <div class="metric-card"><div class="label">Page-Hinkley</div><div class="value" id="page-hinkley">--</div></div>
                    </div>
                    <div class="status-box" id="drift-status">Drift analysis pending.</div>
                </div>

                <div class="panel">
                    <h2>Retention Decision</h2>
                    <div class="section-grid">
                        <div class="metric-card"><div class="label">LinUCB Action</div><div class="value" id="linucb-action">--</div></div>
                        <div class="metric-card"><div class="label">Override</div><div class="value" id="override-status">--</div></div>
                    </div>
                    <div class="metric-card" style="margin-top: 14px;"><div class="label">Final Recommended Action</div><div class="value" id="final-action">--</div></div>
                    <div class="status-box" id="decision-explanation">Awaiting analysis.</div>
                </div>

                <div class="panel">
                    <h2>Customer Profile Visualization</h2>
                    <div class="risk-cards">
                        <div class="metric-card"><div class="label">Total Orders</div><div class="value" id="total-orders">--</div></div>
                        <div class="metric-card"><div class="label">Total Spending</div><div class="value" id="metric-spend">--</div></div>
                        <div class="metric-card"><div class="label">Average Order Value</div><div class="value" id="metric-aov">--</div></div>
                        <div class="metric-card"><div class="label">Recency</div><div class="value" id="metric-recency">--</div></div>
                        <div class="metric-card"><div class="label">Purchase Frequency</div><div class="value" id="metric-frequency">--</div></div>
                        <div class="metric-card"><div class="label">Average Gap</div><div class="value" id="metric-gap">--</div></div>
                        <div class="metric-card"><div class="label">Review Score</div><div class="value" id="metric-review">--</div></div>
                    </div>
                    <div class="status-box" style="margin-top: 14px;">
                        <strong>Purchase Timeline</strong>
                        <div id="purchase-timeline" style="margin-top: 8px; font-weight: 700;">Waiting for customer sequence.</div>
                    </div>
                </div>

                <div class="panel">
                    <h2>Pipeline Progress</h2>
                    <div class="progress"><div class="progress-bar" id="progress-bar"></div></div>
                    <ol class="ordered-list" id="pipeline-steps">
                        <li>Preprocessing customer data...</li>
                        <li>Running Random Forest...</li>
                        <li>Running XGBoost...</li>
                        <li>Generating GRU behaviour representation...</li>
                        <li>Creating 66 fusion features...</li>
                        <li>Generating final churn probability...</li>
                        <li>Detecting behaviour drift...</li>
                        <li>Calculating CRPI...</li>
                        <li>Running LinUCB...</li>
                        <li>Applying recommendation override...</li>
                        <li>Final retention recommendation generated.</li>
                    </ol>
                </div>

                <div class="panel">
                    <h2>Embedding Values</h2>
                    <div class="embeddings" id="embedding-output">No embedding generated yet.</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const demoData = {};
        const defaultForm = {
            customer_id: 'DEMO_02',
            display_name: 'Customer Demo 02',
            risk_profile: 'Medium',
            tenure_months: '39',
            previous_orders: '20',
            avg_order_value: '198.60',
            total_spending: '3972.00',
            recency_days: '52',
            avg_days_between_purchases: '48',
            last_order_amount: '166.00',
            avg_review_score: '3.6',
            cancelled_orders: '5',
            delivered_orders: '15',
            payment_attempts: '23',
            preferred_payment_type: 'Bank Transfer',
            product_count: '14',
            seller_count: '10',
            category: 'Electronics',
            recent_purchase_behavior: 'Recent orders have dropped sharply and the customer now buys less frequently with increasing gaps between orders.',
            previous_purchase_behavior: 'Customer previously purchased regularly but has shown a significant recent regime change in purchase cadence and value.'
        };

        async function loadDemoCustomers() {
            const response = await fetch('/api/demo-customers');
            const data = await response.json();
            data.forEach((customer) => {
                demoData[customer.customer_id] = customer;
            });
            return data;
        }

        function fillForm(customer) {
            Object.entries(defaultForm).forEach(([key, value]) => {
                const input = document.getElementById(key);
                if (!input) return;
                const valueFromCustomer = customer[key] ?? value;
                input.value = valueFromCustomer;
            });
            const riskSelect = document.getElementById('risk_profile');
            riskSelect.value = customer.risk_profile || 'Medium';
            document.getElementById('display_name').value = customer.display_name || customer.customer_id;
        }

        function collectForm() {
            const form = document.getElementById('customer-form');
            const formData = new FormData(form);
            return Object.fromEntries(formData.entries());
        }

        function setProgress(index) {
            const bar = document.getElementById('progress-bar');
            const pct = ((index + 1) / 11) * 100;
            bar.style.width = pct + '%';
        }

        function renderResult(result) {
            document.getElementById('customer-name').textContent = result.display_name || result.customer_id;
            document.getElementById('churn-probability').textContent = (result.final_churn_probability * 100).toFixed(1) + '%';
            document.getElementById('customer-value').textContent = '₹' + Number(result.static_features.total_spending).toLocaleString('en-IN');
            document.getElementById('drift-severity').textContent = Number(result.behaviour_drift_score).toFixed(2);
            document.getElementById('crpi-score').textContent = Number(result.crpi_score).toFixed(2);
            document.getElementById('rf-probability').textContent = (result.rf_probability * 100).toFixed(1) + '%';
            document.getElementById('xgb-probability').textContent = (result.xgb_probability * 100).toFixed(1) + '%';
            document.getElementById('gru-embedding-size').textContent = '64-D Embedding Generated';
            document.getElementById('drift-score').textContent = Number(result.behaviour_drift_score).toFixed(2);
            document.getElementById('page-hinkley').textContent = result.page_hinkley_change_detected;
            document.getElementById('linucb-action').textContent = result.base_linucb_action;
            document.getElementById('override-status').textContent = result.override_triggered ? 'Triggered' : 'Not Triggered';
            document.getElementById('final-action').textContent = result.final_retention_action;
            document.getElementById('drift-status').textContent = 'Drift Classification: ' + result.drift_classification + ' | Page-Hinkley: ' + result.page_hinkley_change_detected;
            document.getElementById('decision-explanation').textContent = 'Offline/Synthetic Reward Simulation: ' + result.base_linucb_action + ' selected with churn, CRPI, drift, and CLV context.';
            document.getElementById('total-orders').textContent = Number(result.static_features.total_orders).toFixed(0);
            document.getElementById('metric-spend').textContent = '₹' + Number(result.static_features.total_spending).toLocaleString('en-IN');
            document.getElementById('metric-aov').textContent = '₹' + Number(result.static_features.avg_order_value).toLocaleString('en-IN');
            document.getElementById('metric-recency').textContent = Number(result.static_features.recency_days).toFixed(0) + ' days';
            document.getElementById('metric-frequency').textContent = Number(result.static_features.total_orders / Math.max(result.static_features.customer_lifetime_days / 30, 1)).toFixed(2) + '/month';
            document.getElementById('metric-gap').textContent = Number((result.static_features.customer_lifetime_days / Math.max(result.static_features.total_orders, 1)) / 30).toFixed(1) + ' months';
            document.getElementById('metric-review').textContent = Number(result.static_features.avg_review_score).toFixed(1);

            const timeline = Array.from({ length: Math.min(8, 8) }, (_, i) => 'Order ' + (i + 1)).join(' → ');
            document.getElementById('purchase-timeline').textContent = timeline;

            const embeddingValues = result.gru_embedding.slice(0, 12).map((v, idx) => 'E' + (idx + 1) + ': ' + Number(v).toFixed(4)).join('  |  ');
            document.getElementById('embedding-output').textContent = embeddingValues + ' ... (' + result.gru_embedding.length + ' values total)';

            document.getElementById('pipeline-status').textContent = 'Artifact status: ' + result.mode + '. ' + result.artifact_status;
        }

        document.getElementById('demo-buttons').addEventListener('click', async (event) => {
            const button = event.target.closest('[data-demo]');
            if (!button) return;
            const customer = demoData[button.dataset.demo];
            if (customer) fillForm(customer);
        });

        document.getElementById('reset-btn').addEventListener('click', () => {
            fillForm(defaultForm);
            document.getElementById('pipeline-status').textContent = 'Status: Ready for analysis.';
            document.getElementById('progress-bar').style.width = '0%';
            document.getElementById('embedding-output').textContent = 'No embedding generated yet.';
        });

        document.getElementById('analyze-btn').addEventListener('click', async () => {
            const payload = collectForm();
            document.getElementById('pipeline-status').textContent = 'Processing pipeline...';
            for (let i = 0; i < 11; i += 1) {
                setProgress(i);
                await new Promise((resolve) => setTimeout(resolve, 220));
            }
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8' },
                    body: new URLSearchParams(payload).toString()
                });
                const result = await response.json();
                renderResult(result);
            } catch (error) {
                document.getElementById('pipeline-status').textContent = 'Error while running pipeline: ' + error.message;
            }
        });

        loadDemoCustomers().then(() => fillForm(demoData['DEMO_02'] || defaultForm));
    </script>
</body>
</html>
"""


def send_json(handler, payload):
    data = json.dumps(payload).encode('utf-8')
    handler.send_response(200)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def build_html() -> bytes:
    return HTML_TEMPLATE.encode('utf-8')


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/':
            body = build_html()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == '/api/demo-customers':
            send_json(self, load_demo_customers())
            return

        self.send_response(404)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(b'{"error":"Not found"}')

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != '/api/analyze':
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get('Content-Length', '0'))
        raw = self.rfile.read(length)
        payload = parse_qs(raw.decode('utf-8'), keep_blank_values=True)
        cleaned = {key: values[0] if values else '' for key, values in payload.items()}
        try:
            analysis = analyze_customer(cleaned)
            send_json(self, analysis)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self.send_response(500)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode('utf-8'))

    def log_message(self, fmt, *args):
        return


def main():
    server = HTTPServer((HOST, PORT), DemoHandler)
    print(f"Adaptive Hybrid Customer Retention demo running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
    main()
