(() => {
  'use strict';

  const state = {
    availability: null,
    jobs: [],
    runs: [],
    currentRun: null,
    currentJobId: null,
    charts: new Map(),
    selectedCandidateId: null,
    pollBusy: false,
  };

  const pageMeta = {
    dashboard: ['仪表盘', '数据、研究门禁与任务状态'],
    daily: ['每日研究', 'v11 Top5 全市场候选与三本账'],
    history: ['运行历史', '不可变版本、失败记录与结果对比'],
    stocks: ['股票画像', '全市场代码与中文名称检索'],
    data: ['数据中心', '真实 Tushare 数据覆盖与同步任务'],
    models: ['模型审计', '模型选择期、最终 OOS 与晋升门禁'],
    factors: ['因子验证', '窗口化 Factor Lab 研究'],
    portfolio: ['持仓观察', '人工资金决策背景下的风险观察'],
    backtests: ['基准回测', '买入持有与规则比较基线'],
    reviews: ['复盘记录', '人工判断与系统标签分账'],
    jobs: ['任务中心', '队列、进度、失败和中断恢复'],
  };

  const $ = selector => document.querySelector(selector);
  const $$ = selector => Array.from(document.querySelectorAll(selector));
  const esc = value => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
  const hasNumber = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
  const num = value => hasNumber(value) ? Number(value).toLocaleString('zh-CN') : '--';
  const score = value => hasNumber(value) ? Number(value).toFixed(3) : '--';
  const pct = value => hasNumber(value) ? `${Number(value) >= 0 ? '+' : ''}${(Number(value) * 100).toFixed(2)}%` : '--';
  const pctRaw = value => hasNumber(value) ? `${Number(value).toFixed(1)}%` : '--';
  const dt = value => value ? String(value).replace('T', ' ').replace('Z', '').slice(0, 19) : '--';
  const statusText = value => ({queued:'排队中',running:'运行中',cancel_requested:'取消中',completed:'已完成',failed:'失败',cancelled:'已取消',interrupted:'已中断'}[value] || value || '--');
  const decisionText = value => ({eligible_stable:'稳健准入',eligible_aggressive:'进攻准入',no_signal:'无信号',diagnostic_eligible_stable:'诊断：稳健准入',diagnostic_eligible_aggressive:'诊断：进攻准入',diagnostic_no_signal:'诊断：无信号'}[value] || value || '--');
  const shortStatusText = value => ({pending:'等待结算',pending_blocked_exit:'卖出受阻，继续等待',entry_untradable:'D+1 无法买入',settled:'已结算'}[value] || value || '尚未到期');
  const shortExitText = value => ({gap_stop:'跳空触发止损',gap_take_profit:'跳空触发止盈',same_bar_stop_first:'同日双触发，按止损优先',stop_loss:'触发止损',take_profit:'触发止盈',terminal_close:'D+5 收盘退出',terminal_close_deferred:'D+5 卖出受阻后延期退出'}[value] || value || '--');

  async function api(url, options = {}) {
    const response = await fetch(url, options);
    const text = await response.text();
    let payload;
    try { payload = text ? JSON.parse(text) : {}; }
    catch (_) { throw new Error(`服务返回了非 JSON 响应（HTTP ${response.status}），请在任务中心查看错误。`); }
    if (!response.ok || payload.success === false) {
      const error = payload.error;
      const message = typeof error === 'object' ? error.message : error;
      throw new Error(message || `请求失败（HTTP ${response.status}）`);
    }
    return payload.data ?? payload;
  }

  function toast(message, type = '') {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    $('#toastRegion').appendChild(el);
    window.setTimeout(() => el.remove(), 5200);
  }

  function badge(text, tone = '') { return `<span class="badge ${tone}">${esc(text)}</span>`; }
  function metric(label, value, note = '') { return `<div><span>${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(note)}</small></div>`; }
  function rowKV(label, value) { return `<div><span>${esc(label)}</span><b>${esc(value)}</b></div>`; }

  function renderIcons() {
    if (window.lucide) window.lucide.createIcons({attrs:{'stroke-width':1.8}});
  }

  function chart(id, option) {
    if (!window.echarts) return null;
    const target = document.getElementById(id);
    if (!target) return null;
    let instance = state.charts.get(id);
    if (!instance) {
      instance = window.echarts.init(target, null, {renderer:'canvas'});
      state.charts.set(id, instance);
    }
    instance.setOption(option, true);
    return instance;
  }

  function resizeCharts() { state.charts.forEach(instance => instance.resize()); }

  function currentPage() {
    const route = location.hash.replace('#', '').split('?')[0];
    return pageMeta[route] ? route : 'dashboard';
  }

  async function navigate(page) {
    if (!pageMeta[page]) page = 'dashboard';
    $$('.page').forEach(section => section.classList.toggle('active', section.dataset.page === page));
    $$('.side-nav a').forEach(link => link.classList.toggle('active', link.dataset.route === page));
    $('#pageTitle').textContent = pageMeta[page][0];
    $('#pageSubtitle').textContent = pageMeta[page][1];
    window.setTimeout(resizeCharts, 0);
    if (page === 'history') await loadHistory();
    if (page === 'data') await loadDataStatus();
    if (page === 'models') await loadModels();
    if (page === 'factors') await loadFactorReadiness();
    if (page === 'portfolio') await loadPortfolio();
    if (page === 'backtests') await loadBacktests();
    if (page === 'reviews') await loadReviews();
    if (page === 'jobs') await refreshJobs();
  }

  async function loadHealth() {
    try {
      const data = await api('/api/v1/health');
      $('#healthPill').className = 'health-pill ok';
      $('#healthPill span').textContent = `数据 ${data.latest_ready_date || '--'}`;
    } catch (error) {
      $('#healthPill').className = 'health-pill error';
      $('#healthPill span').textContent = '服务异常';
    }
  }

  async function loadDashboard() {
    try {
      const data = await api('/api/v1/dashboard');
      state.availability = data.availability;
      const research = data.research_status || {};
      const gate = research.research_gate || {};
      $('#dashboardMetrics').innerHTML = [
        metric('最新完整数据', data.availability.latest_ready_date || '--', `上市股票 ${num(data.availability.listed_stock_count)}`),
        metric('研究门禁', gate.status || '--', research.diagnostic_only ? '仅诊断' : '候选'),
        metric('纸面信号日', `${num(research.paper_signal_days || 0)} / ${num(research.usable_minimum_signal_days || 40)}`, `观察记录 ${num(research.observed_signal_days || 0)} 日`),
        metric('活动任务', num(data.active_jobs.length), data.active_jobs.length ? '请查看任务中心' : '当前空闲'),
      ].join('');
      const blocking = gate.blocking || [];
      $('#researchBanner').className = `status-banner ${gate.status === 'usable' ? 'success' : ''}`;
      $('#researchBanner').textContent = `${gate.status || 'unknown'}：当前模型${research.promotion_allowed ? '允许晋升' : '不允许晋升'}。${blocking.length ? `阻断项：${blocking.join('、')}` : '暂无额外阻断项。'}`;
      renderCoverageChart(data.availability);
      renderMarketGate(data.research_status.market_gate || {});
      renderRunTable('#dashboardRunRows', data.latest_runs || [], false);
      updateJobBadge(data.active_jobs.length);
      if (!$('#dailyTradeDate').value) $('#dailyTradeDate').value = data.availability.latest_ready_date || '';
      if (!$('#dataTradeDate').value) $('#dataTradeDate').value = data.availability.latest_ready_date || '';
    } catch (error) {
      $('#researchBanner').className = 'status-banner error';
      $('#researchBanner').textContent = error.message;
      toast(error.message, 'error');
    }
  }

  function renderCoverageChart(availability) {
    const latest = availability.latest_dates || {};
    const ready = availability.latest_ready_date || '';
    const names = ['行情','每日指标','资金流','复权','涨跌停','指数','因子快照'];
    const keys = ['daily','daily_basic','moneyflow','adj_factor','stk_limit','index_daily','factor_snapshot'];
    chart('coverageChart', {
      animationDuration: 350,
      grid:{left:80,right:22,top:12,bottom:30},
      tooltip:{trigger:'axis',formatter:params => `${params[0].name}<br>${latest[keys[params[0].dataIndex]] || '--'}`},
      xAxis:{type:'value',min:0,max:1,axisLabel:{formatter:v => v ? '已同步' : '缺失'},splitLine:{show:false}},
      yAxis:{type:'category',data:names,axisLine:{show:false},axisTick:{show:false}},
      series:[{type:'bar',data:keys.map(key => latest[key] && latest[key] >= ready ? 1 : 0),barWidth:12,itemStyle:{color:'#16857a',borderRadius:2},label:{show:true,position:'right',formatter:p => latest[keys[p.dataIndex]] || '--',color:'#687785'}}]
    });
  }

  function renderMarketGate(gate) {
    const audit = gate.final_audit || {};
    $('#marketGateSummary').innerHTML = [
      rowKV('研究状态', gate.research_status || gate.status || '--'),
      rowKV('覆盖率', pct(audit.coverage)),
      rowKV('准入日平均净收益', pct(audit.eligible_mean_net_return)),
      rowKV('全日历平均净收益', pct(audit.all_calendar_mean_net_return)),
      rowKV('日历最大回撤', pct(audit.calendar_max_drawdown)),
      rowKV('准入日 / 总日数', `${num(audit.eligible_days)} / ${num(audit.days)}`),
    ].join('');
  }

  async function preflightDaily() {
    const tradeDate = $('#dailyTradeDate').value.trim();
    clearDailyResult();
    if (!/^\d{8}$/.test(tradeDate)) {
      $('#preflightBox').className = 'preflight-box blocked';
      $('#preflightBox').textContent = '请输入 8 位交易日期，例如 20260716。';
      return null;
    }
    try {
      const data = await api(`/api/v1/calendar/availability?trade_date=${encodeURIComponent(tradeDate)}`);
      state.availability = data;
      const counts = data.component_counts || {};
      const parts = ['daily','daily_basic','moneyflow','adj_factor','stk_limit'].map(key => `${key} ${num(counts[key] || 0)}`);
      const ready = data.snapshot_ready;
      $('#preflightBox').className = `preflight-box ${ready ? 'ready' : 'blocked'}`;
      $('#preflightBox').textContent = `${tradeDate}：${data.is_local_trade_day ? '本地交易日' : '本地无行情'}；${parts.join(' · ')}。${ready ? '已满足使用已同步数据的基础条件。' : '组件不完整，请选择“完整同步并运行”。'}`;
      return data;
    } catch (error) {
      $('#preflightBox').className = 'preflight-box blocked';
      $('#preflightBox').textContent = error.message;
      return null;
    }
  }

  async function startDailyRun() {
    const tradeDate = $('#dailyTradeDate').value.trim();
    const mode = $('#dailyExecutionMode').value;
    const availability = await preflightDaily();
    if (!availability) return;
    if (mode === 'skip_sync' && !availability.snapshot_ready) {
      toast('已同步数据不完整，请选择完整同步并运行。', 'error');
      return;
    }
    $('#runDailyBtn').disabled = true;
    try {
      const data = await api('/api/v1/research-runs', {
        method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({trade_date:tradeDate,execution_mode:mode})
      });
      state.currentJobId = data.job.id;
      showDailyJob(data.job);
      toast(data.created ? `任务 ${data.job.id.slice(0,8)} 已创建` : '相同任务已在运行，已恢复进度');
      await refreshJobs();
    } catch (error) {
      toast(error.message, 'error');
    } finally {
      $('#runDailyBtn').disabled = false;
    }
  }

  function showDailyJob(job) {
    $('#dailyJobPanel').hidden = false;
    $('#dailyJobStage').textContent = statusText(job.stage || job.status);
    $('#dailyJobPct').textContent = `${job.progress_pct || 0}%`;
    $('#dailyJobBar').style.width = `${job.progress_pct || 0}%`;
    $('#dailyJobMessage').textContent = job.message || statusText(job.status);
  }

  function clearDailyResult() {
    state.currentRun = null;
    state.selectedCandidateId = null;
    $('#dailyResult').hidden = true;
    $('#dailyEmpty').hidden = false;
    $('#candidateList').innerHTML = '';
    $('#candidateEvidence').innerHTML = '';
    const radar = state.charts.get('candidateRadar');
    if (radar) radar.clear();
  }

  async function loadRun(runId) {
    try {
      const run = await api(`/api/v1/research-runs/${encodeURIComponent(runId)}`);
      state.currentRun = run;
      $('#dailyTradeDate').value = run.trade_date;
      $('#dailyEmpty').hidden = true;
      $('#dailyResult').hidden = false;
      $('#runVersionLabel').textContent = `${run.trade_date} · ${run.version_label} · ${run.execution_mode}`;
      $('#runDecision').textContent = decisionText(run.decision);
      $('#runMeta').textContent = `模型 ${run.model_version || '--'} · 完成 ${dt(run.completed_at)} · 运行 ID ${run.id}`;
      const auditStatuses = Object.values(run.audit || {}).map(item => item?.status).filter(Boolean);
      $('#runBadges').innerHTML = [
        badge(run.status, run.status === 'completed' ? 'green' : 'red'),
        badge(run.diagnostic_only ? 'diagnostic_only' : 'candidate', run.diagnostic_only ? 'amber' : 'green'),
        badge(`门禁 ${run.result_summary?.research_gate_status || '--'}`, 'amber'),
        badge(`漂移 ${run.result_summary?.data_drift_status || '--'}`, run.result_summary?.data_drift_status === 'pass' ? 'green' : 'red'),
        badge(`审计 ${auditStatuses.every(value => value === 'pass') ? 'pass' : auditStatuses.join('/') || '--'}`),
      ].join('');
      renderCandidates(run.candidates || []);
      renderPortfolioLedger(run);
      renderManualLedger(run);
      renderAuditLedger(run);
      switchLedger('research');
      location.hash = '#daily';
      await navigate('daily');
    } catch (error) { toast(error.message, 'error'); }
  }

  function renderCandidates(candidates) {
    const target = $('#candidateList');
    if (!candidates.length) {
      target.innerHTML = '<div class="empty-state small"><strong>该版本没有候选</strong><span>弱信号或门禁阻断日允许保存 no_signal。</span></div>';
      $('#candidateEvidence').innerHTML = '';
      return;
    }
    target.innerHTML = candidates.map(row => `
      <div class="candidate-row" data-candidate-id="${row.id}" tabindex="0">
        <div class="candidate-rank">${row.rank}</div>
        <div class="candidate-stock"><strong>${esc(row.name || row.ts_code)}</strong><small>${esc(row.ts_code)} · ${esc(row.industry || '行业未知')} · ${esc(row.candidate_type)}</small></div>
        <div class="candidate-metric"><span>5日净收益</span><b class="${Number(row.expected_net_return_5d) >= 0 ? 'positive':'negative'}">${pct(row.expected_net_return_5d)}</b></div>
        <div class="candidate-metric"><span>10日达5%</span><b>${pct(row.p_hit_5pct_10d)}</b></div>
        <div class="candidate-metric"><span>10日潜力</span><b>${pct(row.expected_mfe_10d)}</b></div>
        <div class="candidate-metric"><span>10日下行</span><b class="negative">${pct(row.expected_mae_10d)}</b></div>
        <div class="candidate-risks">${esc((row.risk_tags || []).join(' / ') || '无附加风险标签')}</div>
      </div>`).join('');
    $$('.candidate-row').forEach(element => {
      element.addEventListener('click', () => selectCandidate(Number(element.dataset.candidateId)));
      element.addEventListener('keydown', event => { if (event.key === 'Enter') selectCandidate(Number(element.dataset.candidateId)); });
    });
    selectCandidate(candidates[0].id);
  }

  function selectCandidate(candidateId) {
    const run = state.currentRun;
    if (!run) return;
    const row = (run.candidates || []).find(item => item.id === candidateId);
    if (!row) return;
    state.selectedCandidateId = candidateId;
    $$('.candidate-row').forEach(element => element.classList.toggle('active', Number(element.dataset.candidateId) === candidateId));
    const safety = Math.max(0, Math.min(100, 100 - Math.abs(Number(row.expected_mae_10d || 0)) * 500));
    chart('candidateRadar', {
      tooltip:{},
      radar:{indicator:[{name:'稳健分',max:100},{name:'进攻分',max:100},{name:'10日机会',max:100},{name:'上涨潜力',max:100},{name:'下行安全',max:100}],radius:'64%',splitNumber:5,axisName:{color:'#687785'}},
      series:[{type:'radar',data:[{value:[Number(row.stable_score||0)*100,Number(row.aggressive_score||0)*100,Number(row.p_hit_5pct_10d||0)*100,Math.min(100,Number(row.expected_mfe_10d||0)*500),safety],name:row.name||row.ts_code,areaStyle:{color:'rgba(23,105,170,.16)'},lineStyle:{color:'#1769aa'},itemStyle:{color:'#1769aa'}}]}]
    });
    const features = row.evidence?.features || {};
    const groups = row.evidence?.groups || {};
    const drift = row.evidence?.data_drift || {};
    const shortPrediction = row.evidence?.short_execution_prediction || {};
    const shortPolicy = row.evidence?.short_execution_policy || {};
    const outcome1 = (row.outcomes || []).find(item => Number(item.horizon) === 1);
    const shortActual = outcome1?.details?.short_execution || null;
    const outcome20 = (row.outcomes || []).find(item => Number(item.horizon) === 20);
    const groupBlock = (title, values, emptyText) => {
      const entries = Object.entries(values || {});
      return `<div class="evidence-item"><strong>${esc(title)}</strong><span>${entries.length ? entries.map(([key,value])=>`${esc(key.replace('feature_',''))} ${score(value)}`).join(' · ') : esc(emptyText)}</span></div>`;
    };
    $('#candidateEvidence').innerHTML = `
      <div class="evidence-item"><strong>${esc(row.name || row.ts_code)} · 模型分歧 ${score(row.model_disagreement)}</strong><span>综合 ${score(row.overall_score)} · 稳健 ${score(row.stable_score)} · 进攻 ${score(row.aggressive_score)}</span></div>
      <div class="evidence-item ${shortPrediction.available ? 'pass':'risk'}"><strong>D+2 独立预测头：${shortPrediction.available ? '可用':'尚不可用'}</strong><span>${shortPrediction.available ? `预期净收益 ${pct(shortPrediction.expected_net_return)} · 上涨概率 ${pct(shortPrediction.positive_probability)} · 止盈先于止损 ${pct(shortPrediction.tp_before_sl_probability)}` : '当前 v11 没有经过隔离审核的 D+2 执行预测头；不使用 5 日预测代替，不补造数值'}</span></div>
      <div class="evidence-item"><strong>D+2 执行口径</strong><span>目标 ${esc(shortPolicy.entry_time_target || '09:35')} 买入；历史使用 ${esc(shortPolicy.entry_price_source || 'daily_open_proxy')} + 显式滑点代理。D+2 起 +1% 止盈 / -5% 止损；同根日线双触发按止损先发生；D+5 收盘终止，跌停不可卖则延期</span></div>
      <div class="evidence-item ${shortActual?.status === 'settled' ? 'pass' : (shortActual?.status === 'pending_blocked_exit' || shortActual?.status === 'entry_untradable' ? 'risk':'')}"><strong>D+2 执行结算：${esc(shortStatusText(shortActual?.status))}</strong><span>${shortActual ? `净收益 ${pct(shortActual.realized_net_return)} · 锁定日 ${pct(shortActual.locked_day_return)} · 退出 ${esc(shortActual.exit_date || '--')} ${esc(shortExitText(shortActual.exit_reason))} · 最大路径下行 ${pct(shortActual.max_adverse_path)} · 阻塞 ${num((shortActual.blocked_exit_dates || []).length)} 日` : '等待 D+2 至 D+5 的真实日线与涨跌停数据；09:35 分钟成交价目前不可审计'}</span></div>
      <div class="evidence-item"><strong>费用与成交限制</strong><span>${shortActual ? `买卖佣金各 ${pct(shortActual.costs?.buy_commission_rate)} / ${pct(shortActual.costs?.sell_commission_rate)} · 卖出印花税 ${pct(shortActual.costs?.sell_stamp_tax_rate)} · 单边滑点 ${pct(shortActual.costs?.slippage_rate)} · 最低佣金因订单金额未知未计入` : '默认研究压力：买卖佣金各 0.03%，卖出印花税 0.10%，单边滑点 0.05%；最低佣金需真实订单金额后校准'}</span></div>
      <div class="evidence-item"><strong>20日机会/结算</strong><span>${outcome20 ? `净收益 ${pct(outcome20.net_return)} · MFE ${pct(outcome20.mfe_close)} · MAE ${pct(outcome20.mae_close)}` : '尚未结算；当前 v11 工件没有独立的 20 日预测头，不补造预测值'}</span></div>
      ${groupBlock('资金持续性',groups.capital_flow,'该冻结版本未暴露独立资金证据；可参考上方 main_net 因子')}
      ${groupBlock('行业/板块强度',groups.industry,'该冻结版本未暴露独立行业证据；可参考上方 relative_industry_strength')}
      ${groupBlock('公告与结构事件',groups.events,'该冻结版本未暴露公告原文证据，需人工核验公告真实性')}
      ${groupBlock('财报与质量',groups.financials,'该冻结版本未暴露独立财报证据，不能推定财务完整')}
      <div class="evidence-item ${drift.status === 'pass' ? 'pass':'risk'}"><strong>数据漂移：${esc(drift.status || '未提供')}</strong><span>${esc((drift.severe_features || []).join('、') || '无严重漂移因子')}</span></div>
      ${Object.entries(features).map(([key,value]) => `<div class="evidence-item"><strong>${esc(key.replace('feature_',''))}</strong><span>${score(value)}</span></div>`).join('')}
      <div class="form-row"><button class="secondary" data-manual="true"><i data-lucide="check"></i>人工保留</button><button class="secondary danger" data-manual="false"><i data-lucide="x"></i>人工排除</button></div>`;
    $$('#candidateEvidence [data-manual]').forEach(button => button.addEventListener('click', () => saveManual(candidateId, button.dataset.manual === 'true')));
    renderIcons();
  }

  async function saveManual(candidateId, selected) {
    try {
      await api('/api/v1/manual-selections', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({candidate_id:candidateId,selected})});
      toast(selected ? '已记录人工保留' : '已记录人工排除');
      if (state.currentRun) await loadRun(state.currentRun.id);
    } catch (error) { toast(error.message, 'error'); }
  }

  function renderPortfolioLedger(run) {
    const rows = run.paper_ledger?.default_70_30_portfolio || [];
    $('#portfolioLedger').innerHTML = `<table><thead><tr><th>排名</th><th>代码</th><th>名称</th><th>类型</th><th>纸面权重</th><th>预算状态</th><th>执行状态</th></tr></thead><tbody>${rows.length ? rows.map(row => `<tr><td>${num(row.rank)}</td><td>${esc(row.ts_code)}</td><td>${esc(row.name||'')}</td><td>${esc(row.candidate_type)}</td><td>${pct(row.paper_weight)}</td><td>${esc(row.budget_status)}</td><td>${esc(row.execution_status)}</td></tr>`).join('') : '<tr><td colspan="7" class="empty">当前门禁不允许建立默认纸面组合；no_signal 按零收益计入全日历。</td></tr>'}</tbody></table>`;
  }

  function renderManualLedger(run) {
    const rows = (run.candidates || []).filter(row => row.manual_selection?.selected);
    $('#manualLedger').innerHTML = `<table><thead><tr><th>排名</th><th>代码</th><th>名称</th><th>类型</th><th>人工原因</th><th>记录时间</th></tr></thead><tbody>${rows.length ? rows.map(row => `<tr><td>${row.rank}</td><td>${esc(row.ts_code)}</td><td>${esc(row.name||'')}</td><td>${esc(row.candidate_type)}</td><td>${esc(row.manual_selection.reason||'未填写')}</td><td>${dt(row.manual_selection.updated_at)}</td></tr>`).join('') : '<tr><td colspan="6" class="empty">尚未人工保留候选。人工记录不会回写模型标签。</td></tr>'}</tbody></table>`;
  }

  function renderAuditLedger(run) {
    const blocks = [
      ['数据版本', run.data_version], ['股票池/快照/地基门禁', run.audit], ['市场状态', run.market_gate], ['结果摘要', run.result_summary]
    ];
    $('#auditLedger').innerHTML = blocks.map(([title,value]) => `<div class="audit-block"><h3>${esc(title)}</h3><pre>${esc(JSON.stringify(value || {}, null, 2))}</pre></div>`).join('');
  }

  function switchLedger(name) {
    $$('[data-ledger-tab]').forEach(button => button.classList.toggle('active', button.dataset.ledgerTab === name));
    const mapping = {research:'ledgerResearch',portfolio:'ledgerPortfolio',manual:'ledgerManual',audit:'ledgerAudit'};
    $$('.ledger-view').forEach(view => view.classList.toggle('active', view.id === mapping[name]));
    window.setTimeout(resizeCharts, 0);
  }

  async function loadHistory() {
    const tradeDate = $('#historyDate').value.trim();
    $('#historyRows').innerHTML = '<tr><td colspan="10" class="empty">正在读取...</td></tr>';
    try {
      const [runs, metrics] = await Promise.all([
        api(`/api/v1/research-runs?limit=100${tradeDate ? `&trade_date=${encodeURIComponent(tradeDate)}` : ''}`),
        api('/api/v1/research-metrics?horizon=5'),
      ]);
      state.runs = runs;
      renderRunTable('#historyRows', state.runs, true);
      renderHistoryMetrics(metrics);
    } catch (error) {
      $('#historyRows').innerHTML = `<tr><td colspan="10" class="empty">${esc(error.message)}</td></tr>`;
    }
  }

  function renderHistoryMetrics(data) {
    $('#historyMetrics').innerHTML = [
      metric('全日历平均净收益',pct(data.mean_net_return),`${num(data.calendar_days)} 个已结算日`),
      metric('信号日正收益率',pct(data.positive_signal_day_rate),`${num(data.settled_signal_days)} 个信号日`),
      metric('Top5 全亏损率',pct(data.all_loss_rate),`Hit5 ${pct(data.hit_5pct_rate)}`),
      metric('最大回撤',pct(data.max_drawdown),`未结算 ${num(data.unsettled_signal_days)} 日`),
    ].join('');
    const rows = data.daily || [];
    chart('historyEquityChart',{grid:{left:58,right:24,top:18,bottom:42},tooltip:{trigger:'axis'},xAxis:{type:'category',data:rows.map(row=>row.trade_date)},yAxis:{type:'value',scale:true,axisLabel:{formatter:value=>`${((value-1)*100).toFixed(1)}%`}},series:[{name:'累计净值',type:'line',showSymbol:false,data:rows.map(row=>row.equity),lineStyle:{color:'#1769aa'},areaStyle:{color:'rgba(23,105,170,.08)'}}]});
  }

  function renderRunTable(selector, runs, selectable) {
    const target = $(selector);
    const cols = selectable ? 10 : 7;
    if (!runs.length) { target.innerHTML = `<tr><td colspan="${cols}" class="empty">暂无不可变研究版本</td></tr>`; return; }
    target.innerHTML = runs.map(run => selectable ? `
      <tr><td class="check-cell"><input type="checkbox" data-compare-run="${esc(run.id)}"></td><td>${esc(run.trade_date)}</td><td>${esc(run.version_label)}</td><td>${esc(run.pipeline)}</td><td>${esc(run.execution_mode)}</td><td>${esc(statusText(run.status))}</td><td>${esc(decisionText(run.decision))}</td><td>${esc(run.model_version||'--')}</td><td>${dt(run.completed_at||run.created_at)}</td><td><button class="table-action" data-open-run="${esc(run.id)}">查看</button></td></tr>` : `
      <tr><td>${esc(run.trade_date)}</td><td>${esc(run.version_label)}</td><td>${esc(statusText(run.status))}</td><td>${esc(decisionText(run.decision))}</td><td>${esc(run.model_version||'--')}</td><td>${dt(run.completed_at||run.created_at)}</td><td><button class="table-action" data-open-run="${esc(run.id)}">打开</button></td></tr>`).join('');
    $$('[data-open-run]').forEach(button => button.addEventListener('click', () => loadRun(button.dataset.openRun)));
    $$('[data-compare-run]').forEach(input => input.addEventListener('change', updateCompareButton));
  }

  function updateCompareButton() {
    const count = $$('[data-compare-run]:checked').length;
    $('#compareRunsBtn').disabled = count < 2 || count > 4;
  }

  async function compareRuns() {
    const runIds = $$('[data-compare-run]:checked').map(input => input.dataset.compareRun);
    try {
      const data = await api('/api/v1/research-runs/compare', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_ids:runIds})});
      $('#comparePanel').hidden = false;
      $('#comparePanel').innerHTML = data.runs.map(run => `<div class="compare-item"><h3>${esc(run.trade_date)} ${esc(run.version_label)}</h3>${rowKV('状态',statusText(run.status))}${rowKV('决策',decisionText(run.decision))}${rowKV('模型',run.model_version||'--')}${rowKV('Top5',num((run.candidates||[]).length))}${rowKV('数据日期',run.data_version?.latest_ready_date||'--')}${rowKV('完成时间',dt(run.completed_at))}</div>`).join('');
    } catch (error) { toast(error.message, 'error'); }
  }

  async function searchStock(query) {
    query = String(query || '').trim();
    if (!query) return;
    $('#stockSearchInput').value = query;
    location.hash = '#stocks';
    await navigate('stocks');
    $('#stockSearchResults').innerHTML = '<span class="muted">正在检索全市场股票池...</span>';
    $('#stockProfile').hidden = true;
    try {
      const rows = await api(`/api/stocks/search?q=${encodeURIComponent(query)}`);
      if (!rows.length) { $('#stockSearchResults').innerHTML = '<span class="muted">没有找到匹配股票。</span>'; return; }
      $('#stockSearchResults').innerHTML = rows.slice(0,12).map(row => `<button class="search-result" data-stock-code="${esc(row.ts_code)}">${esc(row.name)} · ${esc(row.ts_code)}</button>`).join('');
      $$('[data-stock-code]').forEach(button => button.addEventListener('click', () => loadStockProfile(button.dataset.stockCode)));
      await loadStockProfile(rows[0].ts_code);
    } catch (error) { $('#stockSearchResults').innerHTML = `<span class="negative">${esc(error.message)}</span>`; }
  }

  async function loadStockProfile(tsCode) {
    $('#stockProfile').hidden = true;
    const existing = state.charts.get('stockRadar');
    if (existing) existing.clear();
    try {
      const profile = await api(`/api/stocks/${encodeURIComponent(tsCode)}/profile`);
      const stock = profile.stock || {};
      const scores = profile.scores || {};
      const evidence = profile.evidence || {};
      $('#stockProfile').hidden = false;
      $('#stockCode').textContent = stock.ts_code || tsCode;
      $('#stockName').textContent = stock.name || tsCode;
      $('#stockIndustry').textContent = `${stock.industry || '行业未知'} · ${stock.area || '地区未知'} · 数据 ${profile.as_of_date || '--'}`;
      const active = profile.active_research || {};
      $('#stockBadges').innerHTML = [badge(active.research_gate||'--','amber'),badge(active.diagnostic_only?'diagnostic_only':'candidate',active.diagnostic_only?'amber':'green'),badge(active.model_version||'--')].join('');
      const dimensions = scores.radar || [];
      const radarRows = Array.isArray(dimensions) ? dimensions : Object.entries(dimensions).map(([label,value]) => ({label,value}));
      chart('stockRadar', {tooltip:{},radar:{indicator:radarRows.map(item=>({name:item.label||item.name,max:100})),radius:'66%'},series:[{type:'radar',data:[{value:radarRows.map(item=>Number(item.value||item.score||0)),areaStyle:{color:'rgba(22,133,122,.16)'},lineStyle:{color:'#16857a'},itemStyle:{color:'#16857a'}}]}]});
      const quote = profile.quote || {};
      const valuation = profile.valuation || {};
      const factors = {
        '稳健描述分':scores.stable,'进攻描述分':scores.aggressive,'强势状态':scores.strong_state,
        '强势分':scores.strong_score,'研究匹配状态':scores.research_fit_state,'研究匹配分':scores.research_fit_score,
        '收盘价':quote.close,'当日涨跌幅':quote.pct_chg == null ? null : pctRaw(quote.pct_chg),
        '换手率':valuation.turnover_rate == null ? null : pctRaw(valuation.turnover_rate),'市盈率 TTM':valuation.pe_ttm,'市净率':valuation.pb,
      };
      $('#stockFactorDetails').innerHTML = Object.entries(factors).map(([key,value]) => rowKV(key,value ?? '--')).join('');
      const missing = evidence.missing_factors || [];
      const missingCount = Number(evidence.missing_factor_count || missing.length);
      $('#stockDataDate').textContent = profile.as_of_date || '--';
      $('#stockQuality').innerHTML = `${!profile.available?`<div class="evidence-item risk"><strong>画像不可用</strong><span>${esc(profile.reason||'没有已冻结因子快照')}</span></div>`:''}<div class="evidence-item ${missingCount?'risk':'pass'}"><strong>缺失因子 ${missingCount} 项</strong><span>${esc(missing.join('、') || (missingCount ? '快照只记录了缺失数量' : '当前画像未报告缺失因子'))}</span></div>${(evidence.risk_tags||[]).map(tag=>`<div class="evidence-item risk"><strong>风险</strong><span>${esc(tag)}</span></div>`).join('')}`;
    } catch (error) { toast(error.message,'error'); }
  }

  async function loadDataStatus() {
    const dateValue = $('#dataTradeDate').value.trim();
    try {
      const data = await api(`/api/v1/data/status${dateValue ? `?trade_date=${encodeURIComponent(dateValue)}`:''}`);
      const storage = data.storage || {};
      const availability = data.availability || {};
      if (!$('#dataTradeDate').value) $('#dataTradeDate').value = availability.latest_ready_date || '';
      $('#dataMetrics').innerHTML = [metric('当前上市股票',num(availability.listed_stock_count),'正式动态股票池'),metric('历史股票记录',num(availability.historical_stock_count),'含退市/暂停记录'),metric('行情总行数',num(storage.daily_quote_count),storage.latest_quote_date||'--'),metric('资金流总行数',num(storage.moneyflow_count),storage.latest_moneyflow_date||'--')].join('');
      $('#dataCoverageDate').textContent = availability.trade_date || '请选择日期';
      const counts = availability.component_counts || {};
      const items = [['行情','daily'],['每日指标','daily_basic'],['资金流','moneyflow'],['复权','adj_factor'],['涨跌停','stk_limit'],['指数','index_daily'],['因子快照','factor_snapshot_v2']];
      chart('dataCoverageChart',{grid:{left:80,right:30,top:12,bottom:30},tooltip:{trigger:'axis'},xAxis:{type:'value'},yAxis:{type:'category',data:items.map(i=>i[0])},series:[{type:'bar',data:items.map(i=>counts[i[1]]||0),barWidth:14,itemStyle:{color:'#1769aa'},label:{show:true,position:'right',formatter:p=>num(p.value)}}]});
      $('#latestDataDates').innerHTML = Object.entries(availability.latest_dates||{}).map(([key,value])=>rowKV(key,value||'--')).join('');
      const endpoints = storage.raw_endpoint_rows || {};
      $('#rawEndpointTable').innerHTML = `<table><thead><tr><th>端点</th><th>真实记录数</th><th>状态</th></tr></thead><tbody>${Object.entries(endpoints).sort((a,b)=>a[0].localeCompare(b[0])).map(([key,value])=>`<tr><td>${esc(key)}</td><td>${num(value)}</td><td>${value>0?badge('已同步','green'):badge('缺失','red')}</td></tr>`).join('')}</tbody></table>`;
      await loadHistoricalAvailabilityAudit();
    } catch (error) { toast(error.message,'error'); }
  }

  async function loadHistoricalAvailabilityAudit() {
    const summary = $('#historicalAuditSummary');
    const table = $('#historicalAuditTable');
    try {
      const data = await api('/api/v1/data/availability-audit');
      if (!data.available) {
        summary.innerHTML = `<div class="evidence-item risk"><strong>审计尚未生成</strong><span>${esc(data.message || '暂无报告')}</span></div>`;
        table.innerHTML = '';
        return;
      }
      const report = data.report || {};
      const conclusion = report.conclusion || {};
      const financial = report.financial || {};
      const events = report.events || {};
      const revision = financial.revision_timing_audit || {};
      summary.innerHTML = [
        ['历史财务层', conclusion.historical_financial_layer || '--', 'risk'],
        ['财务修订时点', revision.status === 'blocked' ? '未通过审计，不能入历史训练' : (revision.status || '--'), revision.status === 'blocked' ? 'risk' : 'pass'],
        ['当前 v11 财务因子', conclusion.current_v11_uses_financial_features ? '已使用' : '未使用', conclusion.current_v11_uses_financial_features ? 'risk' : 'pass'],
        ['当前 v11 公告因子', conclusion.current_v11_uses_v8_announcement_features ? '已使用' : '未使用', conclusion.current_v11_uses_v8_announcement_features ? 'risk' : 'pass'],
        ['当前 v11 结构事件', conclusion.current_v11_uses_v9_structural_features ? '已使用' : '未使用', conclusion.current_v11_uses_v9_structural_features ? 'pass' : 'risk'],
        ['全训练期覆盖', `${num(conclusion.full_training_history_dates_audited || 0)} 个冻结交易日`, 'pass'],
        ['下一步', conclusion.next_action || '--', 'risk'],
      ].map(([title,value,kind]) => `<div class="evidence-item ${kind}"><strong>${esc(title)}</strong><span>${esc(value)}</span></div>`).join('');
      const checkpoints = financial.checkpoints || [];
      const fields = [...new Set(checkpoints.flatMap(item => (item.fields || []).map(field => field.field)))];
      if (!fields.length) {
        table.innerHTML = '<div class="empty">报告没有字段覆盖矩阵。</div>';
        return;
      }
      const history = financial.full_training_history || {};
      const blockedFields = (history.field_summary || []).filter(item => item.status !== 'historically_sufficient').map(item => item.field);
      table.innerHTML = `<table><thead><tr><th>财务字段</th>${checkpoints.map(item => `<th>${esc(item.signal_date)}</th>`)}</tr></thead><tbody>${fields.map(field => `<tr><td>${esc(field)}</td>${checkpoints.map(item => { const detail = (item.fields || []).find(value => value.field === field) || {}; const coverage = detail.coverage == null ? '--' : pct(detail.coverage); return `<td>${coverage}<br>${badge(detail.status === 'historically_sufficient' ? '覆盖足够' : '不足', detail.status === 'historically_sufficient' ? 'green' : 'red')}</td>`; }).join('')}</tr>`).join('')}</tbody></table><p class="table-note">全训练期 ${num(history.date_count || 0)} 个冻结交易日：覆盖低于阈值的字段 ${esc(blockedFields.join('、') || '无')}。财务修订时点：${esc(revision.reason || '未提供')}。原始财务行 ${num(financial.raw_row_count || 0)}；时间规则：${esc(report.audit_scope?.financial_and_announcement_policy || '--')}。结构端点 ${num(Object.keys(events.endpoints || {}).length)} 个，离散事件不应被解释为每日全覆盖。</p>`;
    } catch (error) {
      summary.innerHTML = `<div class="evidence-item risk"><strong>审计读取失败</strong><span>${esc(error.message)}</span></div>`;
      table.innerHTML = '';
    }
  }

  async function submitDataJob(action) {
    const tradeDate = $('#dataTradeDate').value.trim();
    try {
      const data = await api('/api/v1/data-jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,trade_date:tradeDate||null})});
      toast(`数据任务 ${data.job.id.slice(0,8)} 已入队`);
      await refreshJobs();
    } catch (error) { toast(error.message,'error'); }
  }

  async function loadModels() {
    try {
      const data = await api('/api/ml/model_report');
      const report = data.report || {};
      const audit = data.latest_research_audit || {};
      const shortResearch = data.latest_short_execution_research || {};
      const shortFinal = shortResearch.final_audit || {};
      const shortRepro = shortResearch.reproducibility_audit || {};
      const shortOos = data.latest_short_execution_oos_audit || {};
      const shortMapped = shortOos.existing_v11_top5 || {};
      const shortGate = shortOos.v10_market_gate || {};
      const recentWindow = data.recent_window_study || {};
      const recentWindowFinal = recentWindow.final_audit || {};
      const failureWindow = data.failure_window_audit || {};
      const failureMarket = failureWindow.market_state_data || {};
      const blind = data.short_execution_blind || {};
      const blindEval = blind.evaluation || {};
      $('#modelGateBanner').className = `status-banner ${audit.status === 'usable'?'success':''}`;
      $('#modelGateBanner').textContent = `中期 v11：${audit.status || 'unknown'}；D+2 独立头：${shortResearch.gate?.status || '尚无报告'}。两者均未允许晋升，模型选择期表现不得当作最终 OOS。`;
      $('#modelAudit').innerHTML = [
        rowKV('中期研究审核',audit.status||'--'),
        rowKV('市场状态研究',data.latest_market_state_research?.gate?.status||'--'),
        rowKV('D+2 独立头',shortResearch.gate?.status||'--'),
        rowKV('D+2 最终 OOS 日均',pct(shortFinal.mean_daily_net_return)),
        rowKV('D+2 正收益折率',pct(shortFinal.positive_fold_rate)),
        rowKV('D+2 连续最大回撤',pct(shortFinal.max_drawdown)),
        rowKV('D+2 GPU 复现审核',shortRepro.status||'--'),
        rowKV('60/90/100/120 对照', recentWindow.available ? `共同隔离 ${num(recentWindowFinal.signal_days || 0)} 日，${recentWindow.gate?.status || '--'}` : '尚未完成'),
        rowKV('相对最佳窗口', recentWindow.best_model?.train_window ? `${recentWindow.best_model.train_window} 日，最终 OOS ${pct(recentWindowFinal.mean_daily_net_return)}` : '--'),
        rowKV('旧 v11 映射日均',pct(shortMapped.mean_daily_net_return)),
        rowKV('v10 短周期覆盖率',pct(shortGate.coverage)),
        rowKV('独立头晋升允许',shortResearch.promotion_allowed === true ? '是':'否'),
        rowKV('失败窗口归因', failureWindow.available ? `${failureWindow.oos_date_range?.start || '--'} 至 ${failureWindow.oos_date_range?.end || '--'}，${failureWindow.conclusion?.status || '--'}` : '尚未运行'),
        rowKV('失败归因市场数据', failureWindow.available ? `${failureMarket.status || '--'}；旧训练帧 ${failureMarket.snapshot_quality?.status || '--'}` : '--'),
        rowKV('100 日冻结盲测', blind.available ? `${blind.split?.blind_signal_dates?.start || '--'} 至 ${blind.split?.blind_signal_dates?.end || '--'}，${blind.status || '--'}` : '尚未完成'),
        rowKV('盲测日均净收益', pct(blindEval.mean_daily_net_return)),
        rowKV('盲测正收益日率', pct(blindEval.positive_signal_day_rate)),
        rowKV('盲测最大回撤', pct(blindEval.max_drawdown)),
        rowKV('盲测特征分布', blind.feature_distribution_audit?.status || '--'),
        rowKV('盲测未结算股票日', num(blind.label_snapshot_audit?.unsettled_rows || 0)),
        rowKV('短周期报告',shortResearch.path||'--'),
      ].join('');
      const models = report.models || [];
      const series = models.slice(0,6).map(model => ({name:`${model.model}/${model.target}`,type:'bar',data:(model.evaluations||[]).map(item=>Number(item.top10?.mean_net_return||0)*100)}));
      const windows = models[0]?.evaluations?.map(item=>item.window) || [];
      chart('modelChart',{tooltip:{trigger:'axis'},legend:{type:'scroll',bottom:0},grid:{left:50,right:20,top:20,bottom:60},xAxis:{type:'category',data:windows},yAxis:{type:'value',axisLabel:{formatter:'{value}%'}},series});
      $('#modelRows').innerHTML = models.map(model=>{const values=(model.evaluations||[]).map(item=>({name:item.window,value:Number(item.top10?.mean_net_return||0)}));const mean=values.length?values.reduce((a,b)=>a+b.value,0)/values.length:0;const sorted=[...values].sort((a,b)=>b.value-a.value);return `<tr><td>${esc(model.model)}</td><td>${esc(model.target)}</td><td class="${mean>=0?'positive':'negative'}">${pct(mean)}</td><td>${esc(sorted[0]?.name||'--')} ${pct(sorted[0]?.value)}</td><td>${esc(sorted.at(-1)?.name||'--')} ${pct(sorted.at(-1)?.value)}</td></tr>`}).join('') || '<tr><td colspan="5" class="empty">暂无模型报告</td></tr>';
      const failureTarget = $('#failureWindowAudit');
      if (!failureWindow.available) {
        failureTarget.innerHTML = '<div class="empty">尚未生成最终隔离失败归因报告。</div>';
      } else {
        const classifications = failureWindow.classifications || {};
        const rows = [
          ...(classifications.market_regime || []).map(row => ['市场状态', row.group, row.rows, row.mean_net_return, row.positive_rate]),
          ...(classifications.candidate_rank_segment || []).map(row => ['榜单位置', row.group, row.rows, row.mean_net_return, row.positive_rate]),
          ...(classifications.model_disagreement || []).map(row => ['模型分歧', row.group, row.rows, row.mean_net_return, row.positive_rate]),
          ...(classifications.industry_concentration || []).map(row => ['行业集中度', row.group, row.rows, row.mean_net_return, row.positive_rate]),
        ];
        failureTarget.innerHTML = `<table><thead><tr><th>维度</th><th>分组</th><th>样本</th><th>平均净收益</th><th>正收益率</th></tr></thead><tbody>${rows.map(row => `<tr><td>${esc(row[0])}</td><td>${esc(row[1])}</td><td>${num(row[2])}</td><td class="${Number(row[3]) >= 0 ? 'positive' : 'negative'}">${pct(row[3])}</td><td>${pct(row[4])}</td></tr>`).join('') || '<tr><td colspan="5" class="empty">报告没有可展示分组</td></tr>'}</tbody></table><p class="table-note">${esc(failureWindow.path || '')}。结论：${esc(failureWindow.conclusion?.next_action || '仅描述，不构成新的交易规则。')}</p>`;
      }
    } catch (error) { $('#modelGateBanner').className='status-banner error'; $('#modelGateBanner').textContent=error.message; }
  }

  async function loadFactorReadiness() {
    try {
      const data = await api('/api/factor/readiness');
      const windows = data.windows || [];
      $('#factorMetrics').innerHTML = [metric('股票历史记录',num(data.stock_count),'不等同当前上市数'),metric('验证窗口',num(windows.length),'固定隔离窗口'),metric('行情样本',num(windows.reduce((sum,row)=>sum+(row.quote_rows||0),0)),'窗口合计'),metric('资金流样本',num(windows.reduce((sum,row)=>sum+(row.moneyflow_rows||0),0)),'真实记录')].join('');
      $('#factorWindows').innerHTML = windows.map(row=>rowKV(row.label,`${row.first_trade_date} - ${row.last_trade_date} · ${num(row.trade_date_count)} 日`)).join('');
      chart('factorChart',{grid:{left:70,right:25,top:15,bottom:35},tooltip:{trigger:'axis'},xAxis:{type:'category',data:windows.map(row=>row.name)},yAxis:{type:'value'},series:[{name:'快照',type:'bar',data:windows.map(row=>row.snapshot_rows||0),itemStyle:{color:'#16857a'}},{name:'资金流',type:'bar',data:windows.map(row=>row.moneyflow_rows||0),itemStyle:{color:'#9a6414'}}]});
    } catch (error) { toast(error.message,'error'); }
  }

  async function startFactorJob() {
    const horizons = $$('.check-group input:checked').map(input=>Number(input.value));
    try {
      const data = await api('/api/v1/factor-jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({horizons})});
      toast(`Factor 任务 ${data.job.id.slice(0,8)} 已入队`);
      await refreshJobs();
    } catch (error) { toast(error.message,'error'); }
  }

  function renderFactorResult(result) {
    const rows = result.summary || [];
    $('#factorRows').innerHTML = rows.length ? rows.map(row=>`<tr><td>${esc(row.factor)}</td><td>${num(row.horizon)}</td><td>${esc(row.overall_verdict)}</td><td>${num(row.usable_windows)}</td><td>${num(row.observe_windows)}</td><td>${esc(row.failed_windows||'')}</td></tr>`).join('') : '<tr><td colspan="6" class="empty">任务完成但没有汇总行</td></tr>';
  }

  async function loadPortfolio() {
    try {
      const [rows, summary] = await Promise.all([api('/api/v1/positions'),api('/api/positions/summary').catch(()=>null)]);
      const positions = Array.isArray(rows) ? rows : (rows.positions || []);
      const totalCost = positions.reduce((sum,row)=>sum+Number(row.cost_value||row.buy_price*row.quantity||0),0);
      const totalValue = positions.reduce((sum,row)=>sum+Number(row.market_value||row.current_price*row.quantity||0),0);
      const totalProfit = totalValue-totalCost;
      $('#portfolioMetrics').innerHTML = [metric('观察标的',num(positions.length),'人工录入'),metric('成本参考',num(totalCost.toFixed(2)),'不用于全市场排序'),metric('市值参考',num(totalValue.toFixed(2)),'最近已同步价格'),metric('浮动盈亏',num(totalProfit.toFixed(2)),pct(totalCost?totalProfit/totalCost:null))].join('');
      $('#positionRows').innerHTML = positions.length ? positions.map(row=>`<tr><td>${esc(row.ts_code)}</td><td>${esc(row.name)}</td><td>${num(row.buy_price)}</td><td>${num(row.current_price)}</td><td>${num(row.quantity)}</td><td class="${Number(row.profit||0)>=0?'positive':'negative'}">${num(row.profit)} / ${pctRaw(row.profit_pct)}</td><td>${esc((row.risks||row.risk_tags||[]).join(' / ')||row.operation_tendency||'观察')}</td><td><button class="table-action danger" data-delete-position="${row.id}">删除</button></td></tr>`).join('') : '<tr><td colspan="8" class="empty">暂无持仓观察</td></tr>';
      $$('[data-delete-position]').forEach(button=>button.addEventListener('click',()=>deletePosition(button.dataset.deletePosition)));
      chart('portfolioChart',{grid:{left:70,right:25,top:20,bottom:40},tooltip:{trigger:'axis'},xAxis:{type:'category',data:positions.map(row=>row.name||row.ts_code)},yAxis:{type:'value'},series:[{type:'bar',data:positions.map(row=>({value:Number(row.profit||0),itemStyle:{color:Number(row.profit||0)>=0?'#16734a':'#b13a45'}}))}]});
      $('#portfolioRisks').innerHTML = positions.length ? positions.map(row=>`<div class="evidence-item ${Number(row.profit_pct||0)<-8?'risk':''}"><strong>${esc(row.name||row.ts_code)}</strong><span>${esc((row.risks||row.risk_tags||[]).join('、')||row.operation_tendency||'暂无显性风险')}</span></div>`).join('') : '<div class="evidence-item">暂无持仓风险记录</div>';
    } catch (error) { toast(error.message,'error'); }
  }

  async function addPosition(event) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const payload = Object.fromEntries(form.entries());
    try { await api('/api/v1/positions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); event.currentTarget.reset(); toast('已添加持仓观察'); await loadPortfolio(); }
    catch (error) { toast(error.message,'error'); }
  }

  async function deletePosition(id) {
    try { await api(`/api/v1/positions/${id}`,{method:'DELETE'}); toast('已删除持仓观察'); await loadPortfolio(); }
    catch (error) { toast(error.message,'error'); }
  }

  async function loadBacktests() {
    try {
      const rows = await api('/api/v1/backtests');
      $('#backtestRows').innerHTML = rows.length ? rows.map(row=>`<tr><td>#${row.id}</td><td>${esc(row.start_date)} - ${esc(row.end_date)}</td><td>${num(row.initial_capital)}</td><td>${num(row.final_value)}</td><td class="${Number(row.total_return)>=0?'positive':'negative'}">${pct(row.total_return)}</td><td class="negative">${pct(row.max_drawdown)}</td><td>${dt(row.created_at)}</td></tr>`).join('') : '<tr><td colspan="7" class="empty">暂无回测</td></tr>';
      if (rows[0]) renderBacktestResult({...rows[0],...(rows[0].metadata||{})});
    } catch (error) { toast(error.message,'error'); }
  }

  async function runBacktest(event) {
    event.preventDefault();
    const payload = Object.fromEntries(new FormData(event.currentTarget).entries());
    try { const result=await api('/api/v1/backtests',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); toast('基准回测完成'); renderBacktestResult(result); await loadBacktests(); }
    catch (error) { toast(error.message,'error'); }
  }

  function renderBacktestResult(result) {
    $('#backtestMetrics').innerHTML=[metric('总收益',pct(result.total_return),'买入持有基线'),metric('最大回撤',pct(result.max_drawdown),'区间峰谷'),metric('夏普',score(result.sharpe_ratio),'若数据可用'),metric('最终价值',num(result.final_value),'研究初始值口径')].join('');
    const curve=result.equity_curve||result.metadata?.equity_curve||[];
    chart('backtestChart',{grid:{left:55,right:20,top:20,bottom:45},tooltip:{trigger:'axis'},xAxis:{type:'category',data:curve.map(row=>row.trade_date||row.date)},yAxis:{type:'value',scale:true},series:[{name:'净值',type:'line',showSymbol:false,data:curve.map(row=>row.value||row.equity),lineStyle:{color:'#1769aa'}}]});
  }

  async function loadReviews() {
    try {
      const rows=await api('/api/v1/reviews?limit=100');
      $('#reviewTimeline').innerHTML=rows.length?rows.map(row=>`<article class="timeline-entry"><time>${esc(row.trade_date||dt(row.created_at).slice(0,10))}</time><div><h3>${esc(row.title)}</h3><p>${esc(row.content)}</p><div class="badges">${badge(row.category||'daily')}${row.ts_code?badge(row.ts_code):''}${(row.tags||[]).map(tag=>badge(tag)).join('')}</div></div><button class="table-action danger" data-delete-review="${row.id}">删除</button></article>`).join(''):'<div class="empty-state small">暂无复盘记录</div>';
      $$('[data-delete-review]').forEach(button=>button.addEventListener('click',()=>deleteReview(button.dataset.deleteReview)));
    } catch (error) { toast(error.message,'error'); }
  }

  async function addReview(event) {
    event.preventDefault();
    const payload=Object.fromEntries(new FormData(event.currentTarget).entries());
    payload.tags=String(payload.tags||'').split(',').map(value=>value.trim()).filter(Boolean);
    try { await api('/api/v1/reviews',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); event.currentTarget.reset(); toast('复盘已保存'); await loadReviews(); }
    catch (error) { toast(error.message,'error'); }
  }

  async function deleteReview(id) {
    try { await api(`/api/v1/reviews/${id}`,{method:'DELETE'}); toast('复盘已删除'); await loadReviews(); }
    catch (error) { toast(error.message,'error'); }
  }

  async function refreshJobs() {
    if (state.pollBusy) return;
    state.pollBusy=true;
    try {
      state.jobs=await api('/api/v1/jobs?limit=100');
      const active=state.jobs.filter(job=>['queued','running','cancel_requested'].includes(job.status));
      if (!state.currentJobId && currentPage()==='daily') {
        const recoverable=active.find(job=>job.job_type==='daily_research');
        if (recoverable) state.currentJobId=recoverable.id;
      }
      updateJobBadge(active.length);
      renderJobs(state.jobs);
      if (state.currentJobId) {
        const job=state.jobs.find(item=>item.id===state.currentJobId) || await api(`/api/v1/jobs/${state.currentJobId}`);
        showDailyJob(job);
        if (job.status==='completed') {
          state.currentJobId=null;
          const runId=job.result?.research_run_id || job.research_run_id;
          if (runId) await loadRun(runId);
        } else if (['failed','cancelled','interrupted'].includes(job.status)) {
          state.currentJobId=null;
          toast(job.error?.message||job.message||`任务${statusText(job.status)}`,'error');
        }
      }
      const completedFactor=state.jobs.find(job=>job.job_type==='factor_evaluate'&&job.status==='completed'&&job.result?.summary);
      if(completedFactor) renderFactorResult(completedFactor.result);
    } catch (error) {
      if (currentPage()==='jobs') toast(error.message,'error');
    } finally { state.pollBusy=false; }
  }

  function renderJobs(jobs) {
    $('#jobRows').innerHTML=jobs.length?jobs.map(job=>{const runId=job.research_run_id||job.result?.research_run_id;return `<tr><td><strong>${esc(job.job_type)}</strong><br><span class="muted">${esc(job.id.slice(0,8))}</span></td><td>${esc(job.trade_date||'--')}</td><td>${esc(statusText(job.status))}</td><td>${esc(job.stage||'--')}<br><span class="muted">${esc(job.message||'')}</span></td><td><span class="inline-progress"><span style="width:${job.progress_pct||0}%"></span></span>${job.progress_pct||0}%</td><td>${dt(job.created_at)}</td><td>${dt(job.completed_at)}</td><td>${['queued','running','cancel_requested'].includes(job.status)?`<button class="table-action danger" data-cancel-job="${job.id}">取消</button>`:''}${runId?`<button class="table-action" data-open-run="${runId}">结果</button>`:''}</td></tr>`}).join(''):'<tr><td colspan="8" class="empty">暂无任务</td></tr>';
    $$('[data-cancel-job]').forEach(button=>button.addEventListener('click',()=>cancelJob(button.dataset.cancelJob)));
    $$('[data-open-run]').forEach(button=>button.addEventListener('click',()=>loadRun(button.dataset.openRun)));
  }

  function updateJobBadge(count) { $('#jobBadge').hidden=!count; $('#jobBadge').textContent=String(count); }
  async function cancelJob(id) { try { await api(`/api/v1/jobs/${id}/cancel`,{method:'POST'}); toast('已提交取消请求'); await refreshJobs(); } catch(error){ toast(error.message,'error'); } }

  function bindEvents() {
    window.addEventListener('hashchange',()=>navigate(currentPage()));
    window.addEventListener('resize',resizeCharts);
    $$('[data-route]').forEach(link=>link.addEventListener('click',()=>window.setTimeout(()=>navigate(currentPage()),0)));
    $('[data-action="refresh-dashboard"]').addEventListener('click',loadDashboard);
    $('#globalSearchBtn').addEventListener('click',()=>searchStock($('#globalStockSearch').value));
    $('#globalStockSearch').addEventListener('keydown',event=>{if(event.key==='Enter')searchStock(event.currentTarget.value)});
    $('#stockSearchBtn').addEventListener('click',()=>searchStock($('#stockSearchInput').value));
    $('#stockSearchInput').addEventListener('keydown',event=>{if(event.key==='Enter')searchStock(event.currentTarget.value)});
    $('#preflightBtn').addEventListener('click',preflightDaily);
    $('#dailyTradeDate').addEventListener('input',clearDailyResult);
    $('#runDailyBtn').addEventListener('click',startDailyRun);
    $$('[data-ledger-tab]').forEach(button=>button.addEventListener('click',()=>switchLedger(button.dataset.ledgerTab)));
    $('#historyRefresh').addEventListener('click',loadHistory);
    $('#compareRunsBtn').addEventListener('click',compareRuns);
    $('#dataRefresh').addEventListener('click',loadDataStatus);
    $$('[data-data-job]').forEach(button=>button.addEventListener('click',()=>submitDataJob(button.dataset.dataJob)));
    $('#modelRefresh').addEventListener('click',loadModels);
    $('#runFactorBtn').addEventListener('click',startFactorJob);
    $('#portfolioRefresh').addEventListener('click',loadPortfolio);
    $('#positionForm').addEventListener('submit',addPosition);
    $('#backtestForm').addEventListener('submit',runBacktest);
    $('#reviewForm').addEventListener('submit',addReview);
    $('#jobsRefresh').addEventListener('click',refreshJobs);
  }

  async function init() {
    renderIcons();
    bindEvents();
    await Promise.allSettled([loadHealth(),loadDashboard(),refreshJobs()]);
    await navigate(currentPage());
    window.setInterval(refreshJobs,2000);
    window.setInterval(loadHealth,30000);
  }

  document.addEventListener('DOMContentLoaded',init);
})();
