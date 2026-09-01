document.addEventListener("DOMContentLoaded", () => {
    const churnForm = document.getElementById("churn-form");
    const probPercentageEl = document.getElementById("prob-percentage");
    const gaugeFillEl = document.getElementById("gauge-fill");
    const riskBadgeEl = document.getElementById("risk-badge");
    const predictionSummaryEl = document.getElementById("prediction-summary");
    const recommendationTextEl = document.getElementById("recommendation-text");
    const historyTableBody = document.getElementById("history-table-body");
    const btnRefreshHistory = document.getElementById("btn-refresh-history");
    const apiStatusBadge = document.getElementById("api-status-badge");

    // Check API Health
    async function checkHealth() {
        try {
            const res = await fetch("/health");
            const data = await res.json();
            if (data.status === "healthy") {
                apiStatusBadge.className = "badge rounded-pill bg-success-subtle text-success border border-success px-3 py-2";
                apiStatusBadge.innerHTML = `<i class="fa-solid fa-circle-dot me-1 pulse"></i> API Online (${data.model_name})`;
            } else {
                apiStatusBadge.className = "badge rounded-pill bg-warning-subtle text-warning border border-warning px-3 py-2";
                apiStatusBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation me-1"></i> Model Not Loaded`;
            }
        } catch (err) {
            apiStatusBadge.className = "badge rounded-pill bg-danger-subtle text-danger border border-danger px-3 py-2";
            apiStatusBadge.innerHTML = `<i class="fa-solid fa-circle-xmark me-1"></i> API Offline`;
        }
    }

    // Update Gauge meter SVG
    function updateGauge(prob) {
        // Max stroke-dasharray is ~188.4
        const maxDash = 188.4;
        const offset = maxDash - (maxDash * prob);
        gaugeFillEl.style.strokeDashoffset = offset;

        // Set color according to risk
        if (prob >= 0.70) {
            gaugeFillEl.style.stroke = "#ef4444"; // Red
        } else if (prob >= 0.40) {
            gaugeFillEl.style.stroke = "#f59e0b"; // Amber/Orange
        } else {
            gaugeFillEl.style.stroke = "#10b981"; // Emerald/Green
        }
    }

    // Submit Prediction Form
    churnForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Extract form values into JSON payload
        const payload = {
            customer_id: document.getElementById("customer_id").value || "CUST-001",
            gender: document.getElementById("gender").value,
            senior_citizen: parseInt(document.getElementById("senior_citizen").value),
            partner: document.getElementById("partner").value,
            dependents: document.getElementById("dependents").value,
            tenure: parseInt(document.getElementById("tenure").value),
            phone_service: document.getElementById("phone_service").value,
            multiple_lines: document.getElementById("multiple_lines").value,
            internet_service: document.getElementById("internet_service").value,
            online_security: document.getElementById("online_security").value,
            online_backup: document.getElementById("online_backup").value,
            device_protection: document.getElementById("device_protection").value,
            tech_support: document.getElementById("tech_support").value,
            streaming_tv: document.getElementById("streaming_tv").value,
            streaming_movies: document.getElementById("streaming_movies").value,
            contract: document.getElementById("contract").value,
            paperless_billing: document.getElementById("paperless_billing").value,
            payment_method: document.getElementById("payment_method").value,
            monthly_charges: parseFloat(document.getElementById("monthly_charges").value),
            total_charges: parseFloat(document.getElementById("total_charges").value)
        };

        predictionSummaryEl.innerHTML = `<i class="fa-solid fa-spinner fa-spin me-1"></i> Running XGBoost inference...`;

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Prediction request failed");
            }

            const data = await response.json();

            // Animate probability number
            const probPercent = data.churn_probability_percent;
            probPercentageEl.innerText = `${probPercent}%`;
            updateGauge(data.churn_probability);

            // Set Risk Level Badge
            if (data.risk_level === "High Risk") {
                riskBadgeEl.className = "badge rounded-pill fs-6 px-4 py-2 bg-danger text-white shadow-sm";
                riskBadgeEl.innerText = "🚨 High Risk of Churn";
            } else if (data.risk_level === "Medium Risk") {
                riskBadgeEl.className = "badge rounded-pill fs-6 px-4 py-2 bg-warning text-dark shadow-sm";
                riskBadgeEl.innerText = "⚠️ Medium Churn Risk";
            } else {
                riskBadgeEl.className = "badge rounded-pill fs-6 px-4 py-2 bg-success text-white shadow-sm";
                riskBadgeEl.innerText = "✅ Low Churn Risk (Loyal)";
            }

            predictionSummaryEl.innerHTML = `Customer <strong>${data.customer_id}</strong> has a <strong>${data.churn_probability_percent}%</strong> likelihood of churn. (Evaluated by ${data.model_used})`;
            recommendationTextEl.innerText = data.recommendation;

            // Refresh history table
            loadHistory();

        } catch (err) {
            predictionSummaryEl.innerHTML = `<span class="text-danger"><i class="fa-solid fa-triangle-exclamation me-1"></i> Error: ${err.message}</span>`;
        }
    });

    // Load History Table from SQLite API
    async function loadHistory() {
        try {
            const res = await fetch("/api/history?limit=10");
            const data = await res.json();
            
            if (data.status === "success" && data.data.length > 0) {
                historyTableBody.innerHTML = data.data.map(row => {
                    let badgeClass = row.risk_level === "High Risk" ? "bg-danger" : (row.risk_level === "Medium Risk" ? "bg-warning text-dark" : "bg-success");
                    return `
                        <tr>
                            <td>#${row.id}</td>
                            <td>${row.timestamp}</td>
                            <td class="fw-semibold text-light">${row.customer_id}</td>
                            <td>${row.tenure} mos</td>
                            <td>${row.contract}</td>
                            <td>$${row.monthly_charges.toFixed(2)}</td>
                            <td>${row.internet_service}</td>
                            <td class="fw-bold">${(row.churn_probability * 100).toFixed(1)}%</td>
                            <td><span class="badge ${badgeClass}">${row.risk_level}</span></td>
                        </tr>
                    `;
                }).join("");
            } else {
                historyTableBody.innerHTML = `<tr><td colspan="9" class="text-center text-muted py-3">No predictions logged yet.</td></tr>`;
            }
        } catch (err) {
            console.error("Failed to load prediction history", err);
        }
    }

    // Demo Preset Buttons
    document.getElementById("btn-load-high-risk").addEventListener("click", () => {
        document.getElementById("customer_id").value = "CUST-HIGH-RISK";
        document.getElementById("tenure").value = 1;
        document.getElementById("gender").value = "Female";
        document.getElementById("senior_citizen").value = "0";
        document.getElementById("partner").value = "No";
        document.getElementById("dependents").value = "No";
        document.getElementById("internet_service").value = "Fiber optic";
        document.getElementById("phone_service").value = "Yes";
        document.getElementById("multiple_lines").value = "No";
        document.getElementById("online_security").value = "No";
        document.getElementById("online_backup").value = "No";
        document.getElementById("tech_support").value = "No";
        document.getElementById("device_protection").value = "No";
        document.getElementById("streaming_tv").value = "Yes";
        document.getElementById("streaming_movies").value = "Yes";
        document.getElementById("contract").value = "Month-to-month";
        document.getElementById("paperless_billing").value = "Yes";
        document.getElementById("payment_method").value = "Electronic check";
        document.getElementById("monthly_charges").value = 98.75;
        document.getElementById("total_charges").value = 98.75;
    });

    document.getElementById("btn-load-low-risk").addEventListener("click", () => {
        document.getElementById("customer_id").value = "CUST-LOYAL-558";
        document.getElementById("tenure").value = 60;
        document.getElementById("gender").value = "Male";
        document.getElementById("senior_citizen").value = "0";
        document.getElementById("partner").value = "Yes";
        document.getElementById("dependents").value = "Yes";
        document.getElementById("internet_service").value = "DSL";
        document.getElementById("phone_service").value = "Yes";
        document.getElementById("multiple_lines").value = "Yes";
        document.getElementById("online_security").value = "Yes";
        document.getElementById("online_backup").value = "Yes";
        document.getElementById("tech_support").value = "Yes";
        document.getElementById("device_protection").value = "Yes";
        document.getElementById("streaming_tv").value = "No";
        document.getElementById("streaming_movies").value = "No";
        document.getElementById("contract").value = "Two year";
        document.getElementById("paperless_billing").value = "No";
        document.getElementById("payment_method").value = "Credit card (automatic)";
        document.getElementById("monthly_charges").value = 62.40;
        document.getElementById("total_charges").value = 3744.00;
    });

    btnRefreshHistory.addEventListener("click", loadHistory);

    // Initial calls
    checkHealth();
    loadHistory();
});
