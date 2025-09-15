(function () {
  "use strict";
  const $ = (s, r=document) => r.querySelector(s);
  const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));

  // Desktop "Features" dropdown
  const featuresBtn = $("#featuresBtn");
  const featuresMenu = $("#featuresMenu");
  if (featuresBtn && featuresMenu) {
    let open = false;
    const show = () => { featuresMenu.classList.remove("hidden"); open = true; document.addEventListener("click", onDoc, { once:true }); };
    const hide = () => { featuresMenu.classList.add("hidden"); open = false; };
    const onDoc = (e) => { if (!featuresMenu.contains(e.target) && e.target !== featuresBtn) hide(); };
    featuresBtn.addEventListener("click", (e) => { e.preventDefault(); open ? hide() : show(); });
  }

  // Mobile drawer
  const mobileMenu = $("#mobileMenu");
  const menuOpen = $("#menuOpen");
  const menuClose = $("#menuClose");
  if (mobileMenu && menuOpen && menuClose) {
    const open = () => mobileMenu.classList.remove("hidden");
    const close = () => mobileMenu.classList.add("hidden");
    menuOpen.addEventListener("click", open);
    menuClose.addEventListener("click", close);
    mobileMenu.addEventListener("click", (e) => { if (e.target === mobileMenu) close(); });
  }

  // Auto fade flash messages
  const fadeFlash = () => {
    const boxes = $$(".max-w-3xl .rounded-xl.border.p-3");
    if (!boxes.length) return;
    setTimeout(() => boxes.forEach(b => b.style.opacity = "0"), 4000);
  };
  document.addEventListener("DOMContentLoaded", fadeFlash);

  // Show selected filename below file inputs
  document.addEventListener("DOMContentLoaded", () => {
    $$('input[type="file"]').forEach((inp) => {
      const hint = document.createElement("div");
      hint.className = "mt-1 text-xs text-zinc-500";
      inp.insertAdjacentElement("afterend", hint);
      inp.addEventListener("change", () => {
        const f = inp.files && inp.files[0];
        hint.textContent = f ? `Selected: ${f.name}` : "";
      });
    });
  });
})();
