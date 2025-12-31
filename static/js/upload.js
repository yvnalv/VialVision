const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const btnPredict = document.getElementById("btnPredict");
const btnClear = document.getElementById("btnClear");
const imgResult = document.getElementById("imgResult");
const predJson = document.getElementById("predJson");
const statusText = document.getElementById("statusText");

fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  fileName.textContent = f ? f.name : "No file selected";
});

btnClear.addEventListener("click", () => {
  fileInput.value = "";
  fileName.textContent = "No file selected";
  imgResult.removeAttribute("src");
  predJson.textContent = "{}";
  statusText.textContent = "";
});

btnPredict.addEventListener("click", async () => {
  const f = fileInput.files?.[0];
  if (!f) {
    alert("Please choose an image first.");
    return;
  }

  const form = new FormData();
  form.append("file", f);

  btnPredict.disabled = true;
  statusText.textContent = "Running detection…";

  try {
    const res = await fetch("/predict", { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      statusText.textContent = data?.error || "Prediction failed.";
      predJson.textContent = JSON.stringify(data, null, 2);
      return;
    }

    imgResult.src = `data:image/jpeg;base64,${data.image_base64}`;
    predJson.textContent = JSON.stringify(data.detections, null, 2);
    statusText.textContent = `Done. Found ${data.detections.length} object(s).`;
  } catch (e) {
    statusText.textContent = "Network/server error.";
  } finally {
    btnPredict.disabled = false;
  }
});
