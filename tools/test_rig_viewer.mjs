import assert from "node:assert/strict";
import test from "node:test";
import "../docs/app/rig-viewer.js";
const {transform, compose, buildScene} = globalThis.CalibrexRig;
const raw = (parent, child, t, q=[0,0,0,1]) => ({parent_frame: parent, child_frame: child, translation_m: t, rotation_quat_xyzw: q});
const edge = (p,c,t,q) => ({transform: raw(p,c,t,q), source: "bag_tf_static"});
const approx = (a,b) => a.forEach((v,i) => assert.ok(Math.abs(v-b[i]) < 1e-10, `${a} != ${b}`));

test("parent-child composition rotates child translations and frame axes", () => {
  const parent=transform(raw("root","a",[1,2,0],[0,0,Math.SQRT1_2,Math.SQRT1_2]));
  const child=transform(raw("a","b",[1,0,0]));
  const world=compose(parent,child);
  approx(world.t,[1,3,0]);
  approx([world.R[0],world.R[3],world.R[6]],[0,1,0]);
  const art={frame_tree: {frames: ["root","a","b"], edges: [edge("a","b",[1,0,0]), edge("root","a",[1,2,0],[0,0,Math.SQRT1_2,Math.SQRT1_2])]}};
  const before=structuredClone(art);
  const scene=buildScene(art);
  approx(scene.components[0].nodes[2].t,[1,3,0]);
  assert.deepEqual(art,before);
});

test("disconnected roots stay separate and names are safe map keys", () => {
  const scene=buildScene({frame_tree:{frames:["a","b","c","__proto__","constructor"],edges:[edge("a","b",[1,0,0]),edge("__proto__","constructor",[0,1,0])]}});
  assert.equal(scene.components.length,3);
  assert.equal(scene.components.find((c)=>c.root==="a").nodes.length,2);
  assert.equal(scene.components.find((c)=>c.root==="__proto__").nodes[1].name,"constructor");
});

test("exported estimates mark inherited prior axes and never draw omitted pair placeholders", () => {
  const artifact={schema_version:"slac.bag_estimate/v0.1",frames:{root:"base",entries:[
    {transform:raw("base","imu",[.1,0,0]),axes_from_prior:["x","y","z"]},
    {transform:raw("imu","lidar",[0,.2,0]),axes_from_prior:[]},
  ],omitted:[{frame:"camera"}]},pairs:[{transform:raw("base","camera",[999,999,999])}]};
  const scene=buildScene(artifact);
  assert.equal(scene.estimated,true);
  assert.equal(scene.omitted,1);
  const nodes=scene.components[0].nodes;
  assert.deepEqual(nodes[1].prior,["x","y","z"]);
  assert.deepEqual(nodes[2].priorChain,["imu: x, y, z"]);
  assert.ok(!nodes.some((n)=>n.name==="camera"));
});

test("invalid quaternions, cycles and duplicate parents do not invent connected poses", () => {
  assert.equal(transform(raw("a","b",[0,0,0],[0,0,0,0])),null);
  assert.equal(transform(raw("a","b",[NaN,0,0])),null);
  assert.equal(transform(raw("a","b",[0,0,0],[0,0,Infinity,1])),null);
  const q=transform(raw("a","b",[0,0,0],[0,0,0,2]));
  approx(q.R,[1,0,0,0,1,0,0,0,1]);
  const cyclic=buildScene({frame_tree:{edges:[edge("a","b",[0,0,0]),edge("b","a",[0,0,0])]}});
  assert.equal(cyclic.components.length,0); assert.equal(cyclic.unresolved,2);
  const duplicate=buildScene({frame_tree:{edges:[edge("a","c",[1,0,0]),edge("b","c",[2,0,0])]}});
  assert.equal(duplicate.invalid,1); assert.equal(duplicate.unresolved,1);
  assert.ok(!duplicate.components.some((c)=>c.nodes.some((n)=>n.name==="c")));
});
