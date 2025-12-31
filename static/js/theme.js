(function () {
  const root = document.documentElement;
  const btn = document.getElementById("themeBtn");
  const icon = document.getElementById("themeIcon");

  function setTheme(theme) {
    root.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
    if (icon) icon.textContent = theme === "dark" ? "🌙" : "☀️";
  }

  const saved = localStorage.getItem("theme") || "dark";
  setTheme(saved);

  if (btn) {
    btn.addEventListener("click", () => {
      const current = root.getAttribute("data-theme") || "dark";
      setTheme(current === "dark" ? "light" : "dark");
    });
  }
})();
