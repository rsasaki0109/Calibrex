import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import "../docs/app/camera-imu-timing.js";
const {validate,evaluate,cumulativeAt}=globalThis.CalibrexTiming;
const read=()=>JSON.parse(fs.readFileSync(new URL("../docs/app/samples/camera_imu_timing.json",import.meta.url),"utf8"));

test("clock shifts match independent existing Python interval metrics on the same real samples",()=>{
  const data=validate(read()),before=JSON.stringify(data);
  assert.equal(data.camera_start_s.length,59);
  for(const ref of data.references){
    const result=evaluate(data,ref.shift_ms);
    assert.equal(result.predicted.length,59);
    assert.ok(Math.abs(result.rmse_dps-ref.rmse_dps)<1e-9);
    result.predicted.forEach((p,i)=>p.forEach((v,k)=>assert.ok(Math.abs(v-ref.predicted_dps[i][k])<1e-9)));
    assert.ok(Math.abs(result.rmse_dps**2-result.axis_rmse_dps.reduce((s,v)=>s+v*v,0))<1e-9);
    assert.equal(result.offset_s,data.saved_offset_s+ref.shift_ms/1000);
  }
  assert.ok(evaluate(data,50).rmse_dps>evaluate(data,0).rmse_dps*2);
  assert.ok(evaluate(data,-50).rmse_dps>evaluate(data,0).rmse_dps*2);
  assert.equal(JSON.stringify(data),before);
  assert.throws(()=>evaluate(data,51),/within/);
  assert.throws(()=>evaluate(data,NaN),/within/);
});
test("sampled cumulative interpolation does not silently clamp unsupported times",()=>{
  const times=[0,1,2],values=[[0,0,0],[1,2,3],[4,8,12]];
  assert.deepEqual(cumulativeAt(.5,times,values),[.5,1,1.5]);
  assert.deepEqual(cumulativeAt(2,times,values),[4,8,12]);
  assert.throws(()=>cumulativeAt(-.01,times,values),/support/);
  assert.throws(()=>cumulativeAt(2.01,times,values),/support/);
  const d=read();
  assert.throws(()=>validate({...d,imu_time_s:[...d.imu_time_s].reverse()}),/Invalid/);
  assert.throws(()=>validate({...d,camera_start_s:[-100,...d.camera_start_s.slice(1)]}),/extrapolation/);
  assert.throws(()=>validate({...d,camera_rate_dps:[[Infinity,0,0],...d.camera_rate_dps.slice(1)]}),/Invalid/);
});
