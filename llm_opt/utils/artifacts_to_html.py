import os
import json
from typing import List
from llm_opt.utils.logging_config import logger
from llm_opt.core.iteration_artifact import IterationArtifact
from datetime import datetime


def to_html(artifacts: List[IterationArtifact], out_path: str):
    """
    Generate an interactive HTML page to visualize the optimization process.

    Args:
        artifacts: List of IterationArtifact objects representing the optimization iterations
        out_path: Path where the HTML file will be saved
    """
    # Prepare data for visualization
    iterations_data = []
    for artifact in artifacts:
        iteration_data = {
            "idx": artifact.idx,
            "c_code": artifact.c_code,
            "prompt": artifact.prompt,
            "response": artifact.response,
            "success": artifact.success,
            "error": artifact.error if artifact.error else "",
        }

        # Add performance data if available
        if artifact.performance_report:
            report = artifact.performance_report
            iteration_data["performance"] = {
                "speedup": report.speedup_medians(),
                "c_median": report.median_c_runtime() * 1000,  # Convert to ms
                "numpy_median": report.median_numpy_runtime() * 1000,  # Convert to ms
                "c_quantiles": report.calculate_c_quantiles(),
                "numpy_quantiles": report.calculate_numpy_quantiles(),
            }
        else:
            iteration_data["performance"] = None

        iterations_data.append(iteration_data)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(out_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create JavaScript file
    js_filename = os.path.splitext(os.path.basename(out_path))[0] + ".js"
    js_path = os.path.join(output_dir, js_filename)

    # Write the JavaScript file
    with open(js_path, "w") as f:
        f.write(
            """
// Store the iteration data
const iterationsData = ITERATIONS_DATA_PLACEHOLDER;
let currentIteration = null;
let speedupChart = null;
let runtimeChart = null;

// Initialize the page
function initPage() {
    populateIterationsList();
    createSpeedupChart();
    createRuntimeChart();
    
    // Show the first iteration by default
    if (iterationsData.length > 0) {
        showIteration(iterationsData[0].idx);
    }
}

// Populate the iterations list in the sidebar
function populateIterationsList() {
    const container = document.getElementById('iterations-list');
    
    iterationsData.forEach(iteration => {
        const item = document.createElement('div');
        item.className = 'sidebar-item';
        item.dataset.idx = iteration.idx;
        
        let statusBadge = '';
        if (iteration.success) {
            statusBadge = '<span class="badge badge-success">Success</span>';
        } else {
            statusBadge = '<span class="badge badge-error">Error</span>';
        }
        
        item.innerHTML = 'Iteration ' + iteration.idx + ' ' + statusBadge;
        
        item.addEventListener('click', () => {
            showIteration(iteration.idx);
        });
        
        container.appendChild(item);
    });
}

// Show details for a specific iteration
function showIteration(idx) {
    // Update active state in sidebar
    document.querySelectorAll('.sidebar-item').forEach(item => {
        if (parseInt(item.dataset.idx) === idx) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    // Find the iteration data
    currentIteration = iterationsData.find(item => item.idx === idx);
    
    if (!currentIteration) return;
    
    // Update the details panel
    const detailsContainer = document.getElementById('iteration-details');
    
    let statusText = currentIteration.success ? 
        '<span class="success">Success</span>' : 
        '<span class="error">Error: ' + currentIteration.error + '</span>';
    
    let performanceHtml = '';
    if (currentIteration.performance) {
        const perf = currentIteration.performance;
        
        // Create a combined quantiles table
        let combinedQuantilesTable = '<table class="combined-quantiles">' +
                                    '<tr><th>Quantile</th><th>C Time (ms)</th><th>NumPy Time (ms)</th><th>Ratio</th></tr>';
        
        // Assuming the quantiles are the same for both implementations
        Object.keys(perf.c_quantiles).forEach(q => {
            const cTime = (perf.c_quantiles[q] * 1000).toFixed(4);
            const numpyTime = (perf.numpy_quantiles[q] * 1000).toFixed(4);
            const ratio = (perf.numpy_quantiles[q] / perf.c_quantiles[q]).toFixed(2);
            combinedQuantilesTable += '<tr>' +
                                     '<td>' + (parseFloat(q) * 100).toFixed(1) + '%</td>' +
                                     '<td>' + cTime + ' ms</td>' +
                                     '<td>' + numpyTime + ' ms</td>' +
                                     '<td>' + ratio + 'x</td>' +
                                     '</tr>';
        });
        combinedQuantilesTable += '</table>';
        
        performanceHtml = 
            '<h3>Performance Metrics</h3>' +
            '<div class="performance-summary">' +
                '<div class="performance-card summary-card">' +
                    '<div class="metric-group">' +
                        '<h4>Speedup: ' + perf.speedup.toFixed(2) + 'x</h4>' +
                    '</div>' +
                    '<div class="metric-group">' +
                        combinedQuantilesTable +
                    '</div>' +
                '</div>' +
            '</div>';
    }
    
    detailsContainer.innerHTML = 
        '<h2>Iteration ' + currentIteration.idx + ' ' + statusText + '</h2>' +
        
        performanceHtml +
        
        '<div class="tab-container">' +
            '<div class="tab-buttons">' +
                '<button class="tab-button active" data-tab="c-code">C Code</button>' +
                '<button class="tab-button" data-tab="prompt">Prompt</button>' +
                '<button class="tab-button" data-tab="response">Response</button>' +
            '</div>' +
            
            '<div id="c-code" class="tab-content active">' +
                '<pre><code class="language-c">' + escapeHtml(currentIteration.c_code) + '</code></pre>' +
            '</div>' +
            
            '<div id="prompt" class="tab-content">' +
                '<pre><code>' + escapeHtml(currentIteration.prompt) + '</code></pre>' +
            '</div>' +
            
            '<div id="response" class="tab-content">' +
                '<pre><code>' + escapeHtml(currentIteration.response) + '</code></pre>' +
            '</div>' +
        '</div>';
    
    // Setup tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.dataset.tab;
            
            // Update active state of buttons
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            button.classList.add('active');
            
            // Show the selected tab
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(tabId).classList.add('active');
        });
    });
    
    // Apply syntax highlighting
    document.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });
}

// Create the speedup chart
function createSpeedupChart() {
    const ctx = document.getElementById('speedupChart').getContext('2d');
    
    // Prepare chart data
    const labels = [];
    const speedups = [];
    
    iterationsData.forEach(iteration => {
        labels.push('Iteration ' + iteration.idx);
        if (iteration.performance) {
            speedups.push(iteration.performance.speedup);
        } else {
            speedups.push(null);
        }
    });
    
    speedupChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Speedup (NumPy/C)',
                data: speedups,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(54, 162, 235, 1)',
                pointRadius: 5,
                pointHoverRadius: 7,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Speedup Progression',
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'Speedup: ' + context.raw.toFixed(2) + 'x';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Speedup (higher is better)'
                    }
                }
            }
        }
    });
}

// Create the runtime chart
function createRuntimeChart() {
    const ctx = document.getElementById('runtimeChart').getContext('2d');
    
    // Prepare chart data
    const labels = [];
    const cRuntimes = [];
    const numpyRuntimes = [];
    
    iterationsData.forEach(iteration => {
        labels.push('Iteration ' + iteration.idx);
        if (iteration.performance) {
            cRuntimes.push(iteration.performance.c_median);    
            numpyRuntimes.push(iteration.performance.numpy_median);
        } else {
            cRuntimes.push(null);
            numpyRuntimes.push(null);
        }
    });
    
    runtimeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'C Implementation',
                    data: cRuntimes,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(75, 192, 192, 1)',
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    tension: 0.1
                },
                {
                    label: 'NumPy Implementation',
                    data: numpyRuntimes,
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(255, 99, 132, 1)',
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Runtime Comparison',
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.raw.toFixed(4) + ' ms';
                        }
                    }
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Runtime (ms, lower is better)'
                    }
                }
            }
        }
    });
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize the page when loaded
document.addEventListener('DOMContentLoaded', initPage);
""".replace(
                "ITERATIONS_DATA_PLACEHOLDER", json.dumps(iterations_data)
            )
        )

    # Create the HTML file
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Optimization Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/languages/c.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .container {{
            display: grid;
            grid-template-columns: 250px 1fr;
            gap: 20px;
        }}
        .sidebar {{
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .sidebar-item {{
            padding: 10px;
            margin-bottom: 5px;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        .sidebar-item:hover {{
            background-color: #e9ecef;
        }}
        .sidebar-item.active {{
            background-color: #007bff;
            color: white;
        }}
        .content {{
            background-color: #fff;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .tab-container {{
            margin-top: 20px;
        }}
        .tab-buttons {{
            display: flex;
            border-bottom: 1px solid #dee2e6;
            margin-bottom: 15px;
        }}
        .tab-button {{
            padding: 10px 15px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 16px;
            position: relative;
            bottom: -1px;
        }}
        .tab-button.active {{
            border-bottom: 2px solid #007bff;
            color: #007bff;
            font-weight: bold;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        pre {{
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 15px;
            overflow: auto;
        }}
        code {{
            font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 14px;
        }}
        .chart-container {{
            height: 400px;
            margin-bottom: 30px;
        }}
        .performance-summary {{
            margin-bottom: 20px;
        }}
        .summary-card {{
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .metric-group {{
            margin-bottom: 15px;
        }}
        .metric-group:last-child {{
            margin-bottom: 0;
        }}
        .combined-quantiles {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        .combined-quantiles th, .combined-quantiles td {{
            padding: 8px 10px;
            text-align: right;
        }}
        .combined-quantiles th:first-child, .combined-quantiles td:first-child {{
            text-align: left;
        }}
        .success {{
            color: #28a745;
            font-weight: bold;
        }}
        .error {{
            color: #dc3545;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        table, th, td {{
            border: 1px solid #dee2e6;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 7px;
            font-size: 12px;
            font-weight: 700;
            line-height: 1;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: 10px;
            margin-left: 5px;
        }}
        .badge-success {{
            background-color: #d4edda;
            color: #155724;
        }}
        .badge-error {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .meta-info {{
            color: #6c757d;
            font-size: 14px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>LLM Optimization Visualization</h1>
    
    <div class="meta-info">
        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        <br>
        Total Iterations: {len(artifacts)}
        <br>
        Successful Iterations: {sum(1 for a in artifacts if a.success)}
    </div>

    <div class="chart-container">
        <canvas id="speedupChart"></canvas>
    </div>
    
    <div class="chart-container">
        <canvas id="runtimeChart"></canvas>
    </div>

    <div class="container">
        <div class="sidebar">
            <h3>Iterations</h3>
            <div id="iterations-list">
                <!-- Iterations will be populated here via JavaScript -->
            </div>
        </div>
        
        <div class="content">
            <div id="iteration-details">
                <!-- Iteration details will be populated here via JavaScript -->
            </div>
        </div>
    </div>
    
    <script src="{js_filename}"></script>
</body>
</html>
"""

    # Write the HTML file
    with open(out_path, "w") as f:
        f.write(html_content)

    logger.info(f"HTML visualization saved to {out_path}")
    logger.info(f"JavaScript file saved to {js_path}")
