export function showToast(message, type = "success", durationMs = 2200) {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;

  container.appendChild(toast);

  requestAnimationFrame(() => {
    toast.classList.add("toast-show");
  });

  const hide = () => {
    toast.classList.remove("toast-show");
    toast.classList.add("toast-hide");

    setTimeout(() => {
      toast.remove();
    }, 250);
  };

  setTimeout(hide, durationMs);
}