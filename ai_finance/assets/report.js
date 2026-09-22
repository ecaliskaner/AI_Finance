for (const figure of document.querySelectorAll('.figure')) {
  const spec = JSON.parse(figure.dataset.chart);
  const svg = figure.querySelector('svg');
  const tip = figure.querySelector('.tooltip');
  const cross = figure.querySelector('.crosshair');
  const dot1 = figure.querySelector('.dot-1');
  const dot2 = figure.querySelector('.dot-2');
  const p1 = figure.querySelector('.series-1');
  const p2 = figure.querySelector('.series-2');
  const fmt = v => v === null ? 'n/a'
    : spec.percent ? (v * 100).toFixed(1) + '%'
    : spec.money ? v.toLocaleString(undefined, {maximumFractionDigits: 0})
    : v.toFixed(2);

  const at = (path, i) => {
    const total = path.getTotalLength();
    const n = spec.dates.length;
    return path.getPointAtLength(total * (i / Math.max(n - 1, 1)));
  };

  svg.addEventListener('pointermove', event => {
    const box = svg.getBoundingClientRect();
    const scale = 1000 / box.width;
    const x = (event.clientX - box.left) * scale;
    const ratio = (x - spec.left) / spec.plotW;
    const i = Math.max(0, Math.min(spec.dates.length - 1,
      Math.round(ratio * (spec.dates.length - 1))));

    const a = at(p1, i), b = at(p2, i);
    cross.setAttribute('x1', a.x); cross.setAttribute('x2', a.x);
    cross.style.display = ''; dot1.style.display = ''; dot2.style.display = '';
    dot1.setAttribute('cx', a.x); dot1.setAttribute('cy', a.y);
    dot2.setAttribute('cx', b.x); dot2.setAttribute('cy', b.y);

    tip.hidden = false;
    tip.innerHTML = spec.dates[i] + '<br>Strategy <b>' + fmt(spec.primary[i]) +
      '</b><br>Buy and hold <b>' + fmt(spec.secondary[i]) + '</b>';
    const leftPx = (a.x / scale);
    tip.style.left = Math.min(Math.max(leftPx - tip.offsetWidth / 2, 0),
      box.width - tip.offsetWidth) + 'px';
  });

  svg.addEventListener('pointerleave', () => {
    tip.hidden = true;
    cross.style.display = 'none';
    dot1.style.display = 'none';
    dot2.style.display = 'none';
  });
}
