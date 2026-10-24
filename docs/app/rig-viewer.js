// Dependency-free orthographic 3D frame viewer. Transforms use p_parent = T_parent_child p_child.
"use strict";
(() => {
  const identity = () => ({ t: [0, 0, 0], R: [1, 0, 0, 0, 1, 0, 0, 0, 1] });
  const vector = (R, v) => [0, 1, 2].map((i) => R[3*i]*v[0] + R[3*i+1]*v[1] + R[3*i+2]*v[2]);
  const plus = (a, b) => a.map((v, i) => v + b[i]);
  function transform(raw) {
    const t = raw?.translation_m, q = raw?.rotation_quat_xyzw;
    if (!Array.isArray(t) || t.length !== 3 || !t.every(Number.isFinite) ||
        !Array.isArray(q) || q.length !== 4 || !q.every(Number.isFinite)) return null;
    const norm = Math.hypot(...q);
    if (!Number.isFinite(norm) || norm < 1e-12) return null;
    const [x, y, z, w] = q.map((v) => v / norm);
    return { t: t.slice(), R: [
      1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
      2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
      2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ] };
  }
  function compose(parent, child) {
    const R = Array.from({length: 9}, (_, n) => {
      const i = Math.floor(n / 3), j = n % 3;
      return [0, 1, 2].reduce((sum, k) => sum + parent.R[3*i+k] * child.R[3*k+j], 0);
    });
    return { R, t: plus(parent.t, vector(parent.R, child.t)) };
  }
  function buildScene(artifact) {
    const estimated = artifact.schema_version === "slac.bag_estimate/v0.1";
    const edges = estimated ? (artifact.frames?.entries || []) : (artifact.frame_tree?.edges || []);
    const named = new Set(estimated ? [] : artifact.frame_tree?.frames || []);
    if (estimated && artifact.frames?.root) named.add(artifact.frames.root);
    const incoming = new Set(), records = new Map(), duplicates = new Set();
    let invalid = 0;
    for (const edge of edges) {
      const raw = edge.transform, parent = raw?.parent_frame, child = raw?.child_frame;
      if (typeof parent !== "string" || typeof child !== "string" || !parent || !child) { invalid++; continue; }
      named.add(parent); named.add(child); incoming.add(child);
      if (records.has(child)) duplicates.add(child);
      const pose = transform(raw);
      if (!pose || parent === child) { invalid++; records.set(child, null); continue; }
      records.set(child, { parent, child, pose, raw, prior: estimated ? (edge.axes_from_prior || []).slice() : [], source: estimated ? "exported estimate" : edge.source });
    }
    for (const child of duplicates) { records.set(child, null); invalid++; }
    const children = new Map();
    for (const record of records.values()) if (record) {
      if (!children.has(record.parent)) children.set(record.parent, []);
      children.get(record.parent).push(record);
    }
    const roots = [...named].filter((frame) => !incoming.has(frame)).sort();
    const components = [], visited = new Set();
    for (const root of roots) {
      const nodes = [{ name: root, parent: null, ...identity(), prior: [], priorChain: [], source: "root" }];
      visited.add(root);
      // Iterative traversal also handles deep trees without consuming the JS stack.
      for (let at = 0; at < nodes.length; at++) {
        const parent = nodes[at];
        for (const edge of children.get(parent.name) || []) {
          if (visited.has(edge.child)) continue;
          const pose = compose(parent, edge.pose);
          if (!pose.t.every(Number.isFinite) || !pose.R.every(Number.isFinite)) { invalid++; continue; }
          const priorChain = [...parent.priorChain];
          if (edge.prior.length) priorChain.push(`${edge.child}: ${edge.prior.join(", ")}`);
          nodes.push({ name: edge.child, parent: edge.parent, ...pose, prior: edge.prior, priorChain, local: edge.raw, source: edge.source });
          visited.add(edge.child);
        }
      }
      components.push({ root, nodes });
    }
    return { estimated, components, invalid, unresolved: [...named].filter((n) => !visited.has(n)).length, omitted: estimated ? (artifact.frames?.omitted || []).length : 0 };
  }

  class RigViewer {
    constructor(host) {
      this.host = host; this.scene = {components: []}; this.yaw = -.7; this.pitch = .55; this.zoom = 1;
      host.innerHTML = `<div class="rig-toolbar"><label>Connected tree <select class="rig-root"></select></label><label>Frame <select class="rig-frame"></select></label><div class="rig-buttons"><button type="button" data-view="iso">3D</button><button type="button" data-view="top">XY</button><button type="button" data-view="front">XZ</button><button type="button" data-zoom="in" aria-label="Zoom in">+</button><button type="button" data-zoom="out" aria-label="Zoom out">−</button><button type="button" data-view="reset">Reset</button></div></div>
        <div class="rig-stage"><canvas tabindex="0" role="img" aria-label="3D sensor frame tree. Drag or use arrow keys to orbit; plus and minus zoom; Home resets."></canvas><div class="rig-legend"><span class="rig-x">X</span><span class="rig-y">Y</span><span class="rig-z">Z</span><span>Frame axes · metres</span></div></div>
        <p class="rig-caption note"></p><p class="rig-detail"></p><p class="note">Drag to orbit · select a frame to inspect · arrow keys to rotate, +/− to zoom, Home to reset. Axis glyph lengths are illustrative; sensor positions share one metric scale.</p>`;
      this.canvas = host.querySelector("canvas"); this.ctx = this.canvas.getContext("2d");
      this.rootSelect = host.querySelector(".rig-root"); this.frameSelect = host.querySelector(".rig-frame");
      this.rootSelect.addEventListener("change", () => { this.populateFrames(); this.reset(); });
      this.frameSelect.addEventListener("change", () => this.draw());
      host.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => {
        const view = button.dataset.view;
        if (view === "reset") this.reset();
        else { [this.yaw, this.pitch] = view === "top" ? [0, Math.PI / 2] : view === "front" ? [0, 0] : [-.7, .55]; this.draw(); }
      }));
      host.querySelectorAll("[data-zoom]").forEach((button) => button.addEventListener("click", () => this.scale(button.dataset.zoom === "in" ? 1.25 : .8)));
      this.canvas.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        this.drag = {x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY};
        this.canvas.setPointerCapture(event.pointerId); this.canvas.focus({preventScroll: true});
      });
      this.canvas.addEventListener("pointermove", (event) => {
        if (!this.drag) return;
        this.yaw += (event.clientX - this.drag.x) * .008;
        this.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.pitch + (event.clientY - this.drag.y) * .008));
        this.drag.x = event.clientX; this.drag.y = event.clientY; this.draw();
      });
      this.canvas.addEventListener("pointerup", (event) => {
        if (this.drag && Math.hypot(event.clientX - this.drag.startX, event.clientY - this.drag.startY) < 5) {
          const rect = this.canvas.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
          const hit = (this.hits || []).filter((p) => Math.hypot(x-p.x, y-p.y) < 18).sort((a,b) => Math.hypot(x-a.x,y-a.y)-Math.hypot(x-b.x,y-b.y))[0];
          if (hit) { this.frameSelect.value = hit.name; this.draw(); }
        }
        this.drag = null;
      });
      this.canvas.addEventListener("pointercancel", () => { this.drag = null; });
      this.canvas.addEventListener("lostpointercapture", () => { this.drag = null; });
      this.canvas.addEventListener("keydown", (event) => {
        const keys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-", "Home"];
        if (!keys.includes(event.key)) return;
        event.preventDefault();
        if (event.key === "Home") { this.reset(); return; }
        if (["+", "=", "-"].includes(event.key)) { this.scale(event.key === "-" ? .8 : 1.25); return; }
        this.yaw += event.key === "ArrowLeft" ? -.15 : event.key === "ArrowRight" ? .15 : 0;
        this.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.pitch + (event.key === "ArrowUp" ? .15 : event.key === "ArrowDown" ? -.15 : 0)));
        this.draw();
      });
      this.observer = new ResizeObserver(() => this.draw()); this.observer.observe(this.canvas);
      this.themeQuery = matchMedia("(prefers-color-scheme: dark)"); this.themeListener = () => this.draw(); this.themeQuery.addEventListener("change", this.themeListener);
    }
    options(select, values) {
      select.replaceChildren(...values.map((name) => { const option = document.createElement("option"); option.value = name; option.textContent = name; return option; }));
      select.disabled = !values.length;
    }
    setArtifact(artifact) {
      this.scene = buildScene(artifact);
      this.options(this.rootSelect, this.scene.components.map((c) => c.root)); this.populateFrames(); this.reset();
      const s = this.scene;
      this.host.querySelector(".rig-caption").textContent = (s.estimated
        ? "Exported estimate. Amber dashed edges contain prior axes that were NOT MEASURED; downstream poses inherit that uncertainty. Omitted frames are not drawn."
        : "Candidate calibration from the frame tree, not a measured result or a quality verdict.") +
        (s.components.length > 1 ? " Disconnected trees are shown separately; their relative placement is unknown." : "") +
        (s.omitted ? ` ${s.omitted} omitted frame(s).` : "") +
        (s.invalid || s.unresolved ? ` ${s.invalid} invalid edge(s), ${s.unresolved} unresolved frame(s) not drawn.` : "");
    }
    component() { return this.scene.components.find((c) => c.root === this.rootSelect.value); }
    populateFrames() { this.options(this.frameSelect, (this.component()?.nodes || []).map((n) => n.name)); }
    reset() { this.yaw = -.7; this.pitch = .55; this.zoom = 1; this.draw(); }
    scale(factor) { this.zoom = Math.max(.3, Math.min(6, this.zoom * factor)); this.draw(); }
    draw() {
      if (!this.ctx) return;
      const width = this.canvas.clientWidth, height = this.canvas.clientHeight;
      if (!width || !height) return;
      const dpr = Math.min(devicePixelRatio || 1, 2);
      this.canvas.width = Math.round(width * dpr); this.canvas.height = Math.round(height * dpr);
      const ctx = this.ctx; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const styles = getComputedStyle(this.host), color = (name) => styles.getPropertyValue(name).trim();
      ctx.fillStyle = color("--bg"); ctx.fillRect(0, 0, width, height);
      const nodes = this.component()?.nodes || [];
      if (!nodes.length) {
        ctx.fillStyle = color("--muted"); ctx.font = "14px system-ui"; ctx.fillText("No connected frame transforms to display.", 20, height / 2);
        this.host.querySelector(".rig-detail").textContent = "No frames exported or available."; this.hits = []; return;
      }
      const mins = [0,1,2].map((i) => Math.min(...nodes.map((n) => n.t[i])));
      const maxs = [0,1,2].map((i) => Math.max(...nodes.map((n) => n.t[i])));
      const center = mins.map((v,i) => v / 2 + maxs[i] / 2);
      const span = Math.max(.25, ...maxs.map((v,i) => v - mins[i]));
      if (!Number.isFinite(span)) {
        ctx.fillStyle = color("--muted"); ctx.font = "14px system-ui";
        ctx.fillText("Coordinate range is too large to render safely.", 20, height / 2);
        this.host.querySelector(".rig-detail").textContent = "Inspect the source artifact for the frame coordinates.";
        this.hits = []; return;
      }
      const camera = (p) => {
        const [x,y,z] = p.map((v,i) => (v-center[i]) / span);
        const a = Math.cos(this.yaw)*x - Math.sin(this.yaw)*y, b = Math.sin(this.yaw)*x + Math.cos(this.yaw)*y;
        return { u: a, v: Math.sin(this.pitch)*b - Math.cos(this.pitch)*z };
      };
      // Fit positions and the display-only axis glyphs, so near-coincident rigs remain legible.
      const bounds = nodes.flatMap((node) => [camera(node.t), ...[0,1,2].map((i) =>
        camera(plus(node.t,[node.R[i],node.R[3+i],node.R[6+i]].map((v) => v*span*.18))))]);
      const minU=Math.min(...bounds.map((p)=>p.u)), maxU=Math.max(...bounds.map((p)=>p.u));
      const minV=Math.min(...bounds.map((p)=>p.v)), maxV=Math.max(...bounds.map((p)=>p.v));
      const screenScale=Math.min((width-100)/Math.max(.25,maxU-minU),(height-100)/Math.max(.25,maxV-minV))*this.zoom;
      const scale=screenScale/span;
      const project = (p) => { const c=camera(p); return {x:width/2+screenScale*(c.u-(minU+maxU)/2),y:height/2+screenScale*(c.v-(minV+maxV)/2)}; };
      const line = (a,b,stroke,dash=[]) => { ctx.strokeStyle=stroke; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke(); ctx.setLineDash([]); };
      // A reference grid in the selected root's XY plane; it is not reconstructed geometry.
      ctx.lineWidth = 1;
      for (let i=-4;i<=4;i++) {
        const v=i*span/4;
        line(project([center[0]+v,center[1]-span,0]),project([center[0]+v,center[1]+span,0]),color("--line"));
        line(project([center[0]-span,center[1]+v,0]),project([center[0]+span,center[1]+v,0]),color("--line"));
      }
      const byName = new Map(nodes.map((n) => [n.name,n]));
      ctx.lineWidth = 2;
      for (const node of nodes) if (node.parent && byName.has(node.parent)) {
        line(project(byName.get(node.parent).t), project(node.t), node.priorChain.length ? color("--warn") : color("--muted"), node.priorChain.length ? [5,5] : []);
      }
      this.hits = [];
      const selected = this.frameSelect.value;
      for (const node of nodes) {
        const p=project(node.t); this.hits.push({...p,name:node.name});
        const axisLength=span*.18;
        ["--rig-x","--rig-y","--rig-z"].forEach((name,i) => {
          const dir = [node.R[i],node.R[3+i],node.R[6+i]].map((v) => v*axisLength);
          line(p,project(plus(node.t,dir)),color(name));
        });
        ctx.beginPath(); ctx.arc(p.x,p.y,node.name===selected?7:4,0,2*Math.PI);
        ctx.fillStyle=node.priorChain.length?color("--warn"):color("--accent"); ctx.fill();
        if (node.name===selected) { ctx.strokeStyle=color("--text"); ctx.lineWidth=2; ctx.stroke(); }
      }
      // Labels may move in screen space; frame origins and axes never do.
      const labelRects=[];
      for (const node of nodes.slice().sort((a,b)=>(b.name===selected)-(a.name===selected))) {
        const p=project(node.t);
        ctx.font=(node.name===selected?"600 ":"")+"12px system-ui";
        const label=node.name.length>35?node.name.slice(0,32)+"…":node.name;
        const labelWidth=ctx.measureText(label).width;
        let rect;
        for (let attempt=0;attempt<24;attempt++) {
          const dx=attempt%2===0?11:-labelWidth-11;
          const dy=-9+Math.ceil(attempt/2)*19;
          const x=Math.max(8,Math.min(width-labelWidth-8,p.x+dx));
          const y=Math.max(16,Math.min(height-12,p.y+dy));
          rect={x:x-3,y:y-13,w:labelWidth+6,h:17};
          if (!labelRects.some((r)=>rect.x<r.x+r.w+3 && rect.x+rect.w+3>r.x && rect.y<r.y+r.h+3 && rect.y+rect.h+3>r.y)) break;
        }
        labelRects.push(rect);
        if (Math.abs(rect.y-p.y)>22) line(p,{x:rect.x,y:rect.y+8},color("--muted"),[2,3]);
        ctx.fillStyle=color("--bg"); ctx.fillRect(rect.x,rect.y,rect.w,rect.h);
        ctx.fillStyle=color("--text"); ctx.fillText(label,rect.x+3,rect.y+13);
      }
      const chosen=byName.get(selected) || nodes[0];
      const fmt = (v) => Number(v.toPrecision(5)).toString();
      const local = chosen.local;
      this.host.querySelector(".rig-detail").textContent = `${chosen.name} · root ${this.rootSelect.value} · position [${chosen.t.map(fmt).join(", ")}] m` +
        (local ? ` · parent ${chosen.parent} · local quaternion xyzw [${local.rotation_quat_xyzw.map(fmt).join(", ")}]` : " · root origin") +
        (chosen.priorChain.length ? ` · NOT MEASURED / from prior: ${chosen.priorChain.join("; ")}` : "") +
        (this.scene.estimated ? " · exported estimate" : " · candidate calibration");
      // Scale bar stays in screen space and uses the same metric scale as positions.
      const nominal=80/scale, decade=Math.pow(10,Math.floor(Math.log10(nominal)));
      const metres=(nominal/decade>=5?5:nominal/decade>=2?2:1)*decade;
      if (Number.isFinite(metres) && metres > 0) {
        line({x:20,y:height-25},{x:20+metres*scale,y:height-25},color("--text"));
        ctx.fillStyle=color("--muted"); ctx.font="11px system-ui"; ctx.fillText(`${fmt(metres)} m`,20,height-34);
      }
    }
    destroy() { this.observer.disconnect(); this.themeQuery.removeEventListener("change", this.themeListener); }
  }
  globalThis.CalibrexRig = { transform, compose, buildScene, RigViewer };
})();
