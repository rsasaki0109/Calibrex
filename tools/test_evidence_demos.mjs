import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import "../docs/app/demo-solver.js";
import "../docs/app/evidence-demos.js";
const read=code=>JSON.parse(fs.readFileSync(new URL(`../docs/app/samples/evidence_${code}.json`,import.meta.url),"utf8"));
const {validate}=globalThis.CalibrexEvidence;
const {LiveFit}=globalThis.CalibrexDemoSolver;

test("all five evidence datasets use finite measured coordinates or channels",()=>{
  for(const code of ["comparison","handeye","gnss","vehicle","radar"]){
    const data=validate(read(code),code);
    assert.throws(()=>validate({...data,capture_kind:"synthetic"},code),/Invalid/);
    const first=data.views[0];
    if(first.kind!=="lines")assert.throws(()=>validate({...data,views:[{...first,clouds:[{points:[[NaN,0,1]]}]}]},code),/Invalid/);
    else assert.throws(()=>validate({...data,views:[{...first,chart:{...first.chart,x:[1]}}]},code),/Invalid/);
  }
  assert.equal(read("comparison").views.length,4);
  assert.equal(read("handeye").views.length,8);
  assert.equal(read("vehicle").views.length,5);
});
test("live 6-DoF fit performs real updates and never fits held-out centres",()=>{
  const problem=read("comparison").live_problem;
  const fit=new LiveFit(problem),initial=fit.history[0].fit;
  while(fit.step()){}
  assert.equal(fit.status,"converged");assert.ok(fit.iteration>=2);
  assert.ok(fit.history.at(-1).fit<initial/10);
  assert.ok(fit.history.at(-1).holdout<.03);
  for(let i=1;i<fit.history.length;i++)assert.ok(fit.history[i].fit<=fit.history[i-1].fit+1e-12);
  const changed=structuredClone(problem);
  changed.ids.forEach((id,i)=>{if(!problem.fit_ids.includes(id))changed.target[i]=changed.target[i].map(v=>v+100);});
  const other=new LiveFit(changed);while(other.step()){}
  fit.state.forEach((v,i)=>assert.ok(Math.abs(v-other.state[i])<1e-10));
  assert.ok(other.history.at(-1).holdout>100);
});
