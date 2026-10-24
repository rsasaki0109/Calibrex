import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import "../docs/app/alignment-demo.js";
const {validate,basis,project}=globalThis.CalibrexAlignment;
const data=JSON.parse(fs.readFileSync(new URL("../docs/app/samples/alignment_demo.json",import.meta.url),"utf8"));

test("real-data sample has finite paired endpoints, attribution and digest provenance",()=>{
  assert.equal(validate(data),data);
  assert.equal(data.before.length,data.after.length);
  assert.equal(data.dataset.name,"NTU VIRAL");
  assert.equal(data.dataset.license,"CC BY-NC-SA 4.0");
  assert.ok(data.provenance.comparison.includes("not a live solve"));
  assert.ok(data.provenance.sources.every((s)=>/^[a-f0-9]{64}$/.test(s.sha256)));
  assert.equal(data.metrics.start_residual_m,0.19102774576222076);
  assert.equal(data.metrics.estimate_residual_m,0.04398428152563687);
  assert.notDeepEqual(data.before,data.after);
});
test("projection uses a common camera and rejects points behind it",()=>{
  const camera={target:[0,0,0],azimuth_deg:0,elevation_deg:0,distance_m:10};
  const b=basis(camera);
  assert.deepEqual(b.eye,[10,0,0]);
  const visible=project([0,0,0,20,0,0],camera,800,500,1);
  assert.equal(visible.length,1);
  assert.equal(visible[0].u,400);assert.equal(visible[0].v,250);
  assert.equal(visible[0].depth,10);
  const reference=project(data.reference,data.camera,900,460,0);
  const before=project(data.before,data.camera,900,460,1);
  const after=project(data.after,data.camera,900,460,1);
  assert.ok(reference.length>1000 && before.length>1000 && after.length>1000);
});
test("invalid sample payloads are rejected rather than turned into plausible pictures",()=>{
  assert.throws(()=>validate({...data,before:[NaN,0,0]}),/Invalid sample points/);
  assert.throws(()=>validate({...data,before:[0,0,0]}),/Invalid sample camera/);
  assert.throws(()=>validate({...data,camera:{...data.camera,distance_m:0}}),/Invalid sample camera/);
  assert.throws(()=>validate({...data,metrics:{start_residual_m:Infinity}}),/Invalid sample metrics/);
});
