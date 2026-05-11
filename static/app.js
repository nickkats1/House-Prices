const form = document.getElementById("predict-form");
const result = document.getElementById("result");
const priceEl = document.getElementById("price-value");
const modelEl = document.getElementById("model-used");
const errorEl = document.getElementById("error");

const fmt = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.hidden = true;
  result.hidden = true;

  const data = Object.fromEntries(new FormData(form).entries());

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || "Request failed");

    priceEl.textContent = fmt.format(body.predicted_price);
    modelEl.textContent = `Model: ${body.model}`;
    result.hidden = false;
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  }
});