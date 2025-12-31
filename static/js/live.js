const mjpeg = document.getElementById("mjpeg");
const meta = document.getElementById("streamMeta");
const btnReload = document.getElementById("btnReload");
const btnSnapshot = document.getElementById("btnSnapshot");

function reloadStream() {
  if (!mjpeg) return;
  const url = new URL(mjpeg.src, window.location.href);
  url.searchParams.set("ts", Date.now().toString());
  mjpeg.src = url.toString();
  if (meta) meta.textContent = "Reloaded.";
}

if (btnReload) btnReload.addEventListener("click", reloadStream);

if (mjpeg) {
  mjpeg.onload = () => { if (meta) meta.textContent = "Streaming…"; };
  mjpeg.onerror = () => { if (meta) meta.textContent = "Stream error. Check camera."; };
}

if (btnSnapshot) {
  btnSnapshot.addEventListener("click", async () => {
    try {
      const res = await fetch("/camera/snapshot");
      if (!res.ok) throw new Error("snapshot failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      // Quick open image in new tab (works well on desktop; on Pi LCD it will still show)
      window.open(url, "_blank");
    } catch (e) {
      if (meta) meta.textContent = "Snapshot error.";
    }
  });
}
