import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import "../docs/app/method-gallery.js";
const {METHODS,validateStereo}=globalThis.CalibrexMethods;
const read=(p)=>JSON.parse(fs.readFileSync(new URL("../docs/app/samples/"+p,import.meta.url),"utf8"));

test("gallery exposes five distinct methods and recorded sources remain attributable",()=>{
  assert.equal(METHODS.length,5);assert.equal(new Set(METHODS.map(m=>m.id)).size,5);
  for(const m of METHODS.filter(m=>m.image)) {
    assert.ok(m.source.startsWith("../assets/"));assert.ok(m.license);assert.ok(m.dataset);
    assert.ok(fs.existsSync(new URL("../docs/"+m.image.slice(3),import.meta.url)));
  }
});
test("each native stereo profile has valid images and finite metric receipts",()=>{
  const index=read("fisheye_stereo_demo.json");
  assert.equal(index.schema,"calibrex.fisheye_stereo_demo/v1");
  assert.deepEqual(index.scene.depth_planes_m,[3,6,12]);
  assert.ok(index.method.includes("No truth depths enter matching"));
  for(const code of ["kb4","mei","ds"]){
    const model=validateStereo(read(index.models[code].url));
    assert.ok(model.matches>20);assert.equal(model.baseline_m,.3);
    assert.ok(model.vertical_median_px<.2);assert.ok(model.median_relative_depth_error<.03);
    assert.throws(()=>validateStereo({...model,raw:"javascript:alert(1)"}),/Invalid/);
    assert.throws(()=>validateStereo({...model,vertical_median_px:NaN}),/Invalid/);
    assert.throws(()=>validateStereo({...model,matches:-1}),/Invalid/);
    assert.equal(model.correspondences.length,model.matches);
    const point=model.correspondences[0];
    for(const change of [
      {left:[-1,20]}, {right:[320,20]}, {disparity_px:0},
      {depth_m:point.depth_m+1}, {right:[point.right[0]+1,point.right[1]]},
    ])assert.throws(()=>validateStereo({...model,correspondences:[{...point,...change},...model.correspondences.slice(1)]}),/Invalid/);
    assert.throws(()=>validateStereo({...model,correspondences:[]}),/Invalid/);
  }
});
