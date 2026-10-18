/*
 * Downscale display equations that are wider than the content column.
 *
 * Furo puts display math in an `overflow-x: auto` wrapper, so an equation that
 * overshoots the column has to be scrolled. Scaling it down by exactly the
 * overshoot factor reads better for the occasional wide equation (a group
 * matrix, say) at the cost of slightly smaller glyphs.
 *
 * Equations that already fit are left untouched. The factor is clamped at
 * MIN_SCALE so a pathologically wide equation stays legible -- past that point
 * the wrapper's horizontal scroll takes over again.
 */
(function () {
  "use strict";

  var MIN_SCALE = 0.6;
  var canZoom = window.CSS && CSS.supports && CSS.supports("zoom", "0.5");

  function fit(math) {
    var container = math.querySelector("mjx-container, .MathJax");
    if (!container) {
      return; // not typeset (yet)
    }

    // Reset before measuring, so a widening viewport can scale back up.
    container.style.removeProperty("zoom");
    container.style.removeProperty("transform");
    container.style.removeProperty("transform-origin");
    container.style.removeProperty("display");
    math.style.removeProperty("height");

    // mjx-container is a full-width block; the real width is on the math node.
    var inner = container.querySelector("mjx-math, svg") || container;
    var needed = inner.getBoundingClientRect().width;
    var available = math.clientWidth;
    if (!needed || !available || needed <= available) {
      return;
    }

    var scale = Math.max(MIN_SCALE, available / needed);
    if (canZoom) {
      container.style.zoom = scale;
    } else {
      // transform does not shrink the layout box, so reclaim the height by hand
      var height = container.getBoundingClientRect().height;
      container.style.display = "inline-block";
      container.style.transformOrigin = "left top";
      container.style.transform = "scale(" + scale + ")";
      math.style.height = height * scale + "px";
    }
  }

  function fitAll() {
    var blocks = document.querySelectorAll("div.math");
    for (var i = 0; i < blocks.length; i++) {
      fit(blocks[i]);
    }
  }

  // MathJax is loaded asynchronously, so it may not exist yet; wait for its
  // startup promise (resolved after the initial typesetting) if it shows up.
  var waited = 0;
  function whenTypeset(callback) {
    var startup = window.MathJax && window.MathJax.startup;
    if (startup && startup.promise) {
      startup.promise.then(callback);
    } else if (waited < 10000) {
      waited += 100;
      setTimeout(function () {
        whenTypeset(callback);
      }, 100);
    } else {
      callback(); // no MathJax on this page
    }
  }

  var resizeTimer = null;
  function onResize() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(fitAll, 150);
  }

  function init() {
    whenTypeset(fitAll);
    // Metric fonts can land after typesetting and change the widths.
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(fitAll);
    }
    window.addEventListener("resize", onResize);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
