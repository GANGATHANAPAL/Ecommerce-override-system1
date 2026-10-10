from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from app.inference.model_loader import get_missing_artifacts
from app.inference.pipeline import (
    analyze_customer,
    find_demo_customer,
    get_dataset_overview,
    get_observation_customer,
    load_demo_customers,
    search_observation_customers,
)

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8000"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#111b2b">
  <title>Adaptive Hybrid Customer Retention System</title>
  <style>
    :root {
      color-scheme: light;
      --navy: #111b2b;
      --navy-soft: #1a2940;
      --ink: #17243a;
      --muted: #728096;
      --faint: #9aa5b5;
      --line: #e5eaf1;
      --canvas: #f5f7fa;
      --white: #fff;
      --blue: #4263eb;
      --blue-soft: #edf1ff;
      --violet: #7357d9;
      --green: #188a69;
      --green-soft: #e8f6f0;
      --amber: #b96d18;
      --amber-soft: #fff4e5;
      --red: #bd4855;
      --red-soft: #fff0f1;
      --shadow: 0 8px 24px rgba(27, 42, 68, .045);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--canvas); color: var(--ink); font-size: 14px; }
    button, input, select { font: inherit; }
    button { cursor: pointer; }
    button:focus-visible, input:focus-visible, a:focus-visible { outline: 3px solid #a8b7ff; outline-offset: 2px; }
    .app-shell { min-height: 100vh; }
    .sidebar { position: fixed; z-index: 30; inset: 0 auto 0 0; width: 252px; display: flex; flex-direction: column; color: #e9eef8; background: var(--navy); transition: transform .22s ease; }
    .brand { padding: 25px 21px 23px; border-bottom: 1px solid rgba(255,255,255,.08); }
    .brand-row { display: flex; align-items: center; gap: 11px; }
    .brand-mark { width: 36px; height: 36px; display: grid; place-items: center; color: white; border-radius: 11px; background: #405ee8; }
    .brand-mark svg { width: 21px; height: 21px; }
    .brand-name { font-weight: 720; font-size: 13px; line-height: 1.35; letter-spacing: -.01em; }
    .brand-caption { margin: 13px 0 0 47px; color: #8795ab; font-size: 10px; letter-spacing: .12em; text-transform: uppercase; }
    .nav-section { margin: 22px 12px 8px; color: #728098; font-size: 10px; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; }
    .nav-links { padding: 0 10px; display: grid; gap: 4px; }
    .nav-link { width: 100%; display: flex; align-items: center; gap: 12px; min-height: 42px; padding: 0 12px; border: 0; border-radius: 8px; background: transparent; color: #aab6c8; text-align: left; transition: color .15s, background .15s; }
    .nav-link:hover { color: white; background: rgba(255,255,255,.06); }
    .nav-link.active { color: white; background: #25334c; box-shadow: inset 3px 0 #6e86ff; }
    .nav-link svg { width: 17px; height: 17px; flex: 0 0 17px; opacity: .92; }
    .sidebar-bottom { margin-top: auto; padding: 16px 17px 20px; border-top: 1px solid rgba(255,255,255,.08); }
    .service-card { padding: 12px; border: 1px solid rgba(255,255,255,.09); border-radius: 9px; background: rgba(255,255,255,.035); }
    .service-top { display: flex; align-items: center; gap: 8px; color: #dce4ef; font-size: 11px; font-weight: 650; }
    .service-dot { width: 7px; height: 7px; flex: 0 0 7px; border-radius: 50%; background: #e0a74a; }
    .service-dot.ready { background: #42bd91; box-shadow: 0 0 0 3px rgba(66,189,145,.13); }
    .service-dot.error { background: #ed777e; }
    .service-note { margin: 7px 0 0 15px; color: #8290a6; font-size: 10px; line-height: 1.5; }
    .sidebar-footer { padding: 13px 3px 0; color: #66758b; font-size: 10px; }
    .main { min-height: 100vh; margin-left: 252px; }
    .topbar { height: 70px; position: sticky; z-index: 20; top: 0; display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 0 34px; background: rgba(255,255,255,.94); border-bottom: 1px solid var(--line); backdrop-filter: blur(10px); }
    .top-left { display: flex; align-items: center; gap: 13px; min-width: 0; }
    .menu-toggle { display: none; }
    .breadcrumb { color: var(--muted); font-size: 12px; white-space: nowrap; }
    .breadcrumb strong { color: var(--ink); font-weight: 650; }
    .top-actions { display: flex; align-items: center; gap: 9px; }
    .lookup { width: min(260px, 28vw); display: flex; align-items: center; gap: 8px; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; background: #fafbfd; color: var(--muted); }
    .lookup svg { width: 15px; height: 15px; flex: 0 0 15px; }
    .lookup input { min-width: 0; width: 100%; border: 0; outline: 0; background: transparent; color: var(--ink); font-size: 12px; }
    .icon-button, .button { display: inline-flex; align-items: center; justify-content: center; gap: 8px; border: 1px solid var(--line); border-radius: 8px; background: var(--white); color: #46536a; transition: background .15s, border-color .15s, transform .15s; }
    .icon-button { width: 36px; height: 36px; }
    .icon-button svg { width: 16px; height: 16px; }
    .icon-button:hover, .button:hover { background: #f7f8fb; border-color: #cfd7e3; }
    .button { min-height: 36px; padding: 0 13px; font-size: 12px; font-weight: 650; }
    .button.primary { border-color: var(--blue); background: var(--blue); color: white; }
    .button.primary:hover { border-color: #3555dd; background: #3555dd; }
    .button:disabled { cursor: not-allowed; opacity: .65; }
    .avatar { width: 31px; height: 31px; display: grid; place-items: center; border: 1px solid #dce4ff; border-radius: 50%; background: #eff2ff; color: #455fca; font-size: 11px; font-weight: 750; }
    .content { max-width: 1440px; margin: 0 auto; padding: 30px 34px 52px; }
    .page { display: none; animation: appear .18s ease-out; }
    .page.active { display: block; }
    @keyframes appear { from { opacity: .4; transform: translateY(3px); } to { opacity: 1; transform: translateY(0); } }
    .page-heading { display: flex; align-items: flex-end; justify-content: space-between; gap: 18px; margin-bottom: 23px; }
    .eyebrow { margin-bottom: 7px; color: var(--blue); font-size: 10px; font-weight: 760; letter-spacing: .14em; text-transform: uppercase; }
    h1 { margin: 0; color: var(--ink); font-size: clamp(23px, 2.5vw, 29px); font-weight: 730; letter-spacing: -.035em; line-height: 1.2; }
    .page-description { max-width: 690px; margin: 8px 0 0; color: var(--muted); font-size: 13px; line-height: 1.65; }
    .heading-actions { display: flex; flex-wrap: wrap; gap: 8px; }
    .grid { display: grid; gap: 16px; }
    .kpi-grid { grid-template-columns: repeat(4, minmax(0,1fr)); margin-bottom: 17px; }
    .two-col { grid-template-columns: repeat(2, minmax(0,1fr)); }
    .three-col { grid-template-columns: repeat(3, minmax(0,1fr)); }
    .card { min-width: 0; border: 1px solid var(--line); border-radius: 11px; background: var(--white); box-shadow: var(--shadow); }
    .kpi { min-height: 130px; padding: 17px 18px; }
    .kpi-label { display: flex; align-items: center; justify-content: space-between; gap: 8px; color: var(--muted); font-size: 11px; font-weight: 600; }
    .kpi-icon { width: 29px; height: 29px; display: grid; place-items: center; border-radius: 8px; color: var(--blue); background: var(--blue-soft); }
    .kpi-icon.violet { color: var(--violet); background: #f1edff; }
    .kpi-icon.green { color: var(--green); background: var(--green-soft); }
    .kpi-icon.amber { color: var(--amber); background: var(--amber-soft); }
    .kpi-icon svg { width: 15px; height: 15px; }
    .kpi-value { margin-top: 13px; color: var(--ink); font-size: 27px; font-weight: 740; letter-spacing: -.04em; }
    .kpi-foot { margin-top: 4px; color: var(--faint); font-size: 10px; }
    .card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; padding: 17px 19px 12px; }
    .card-title { margin: 0; font-size: 13px; font-weight: 690; letter-spacing: -.01em; }
    .card-subtitle { margin: 5px 0 0; color: var(--muted); font-size: 10px; line-height: 1.5; }
    .card-body { padding: 8px 19px 18px; }
    .chart { min-height: 160px; }
    .bar-list { display: grid; gap: 13px; }
    .bar-row { display: grid; grid-template-columns: minmax(105px, 1fr) minmax(110px, 2fr) 36px; gap: 10px; align-items: center; color: #59677d; font-size: 11px; }
    .bar-track { height: 8px; overflow: hidden; border-radius: 999px; background: #eff2f6; }
    .bar-fill { height: 100%; min-width: 0; border-radius: inherit; background: var(--blue); transition: width .35s ease; }
    .bar-fill.violet { background: #876ee4; }
    .bar-fill.green { background: #48a98b; }
    .bar-fill.amber { background: #dda34c; }
    .bar-fill.red { background: #d56c75; }
    .bar-count { color: var(--ink); text-align: right; font-weight: 650; font-variant-numeric: tabular-nums; }
    .chart-axis { display: flex; justify-content: space-between; margin: 12px 46px 0 0; color: var(--faint); font-size: 9px; }
    .empty-chart { min-height: 132px; display: grid; place-items: center; padding: 18px; border: 1px dashed #dbe1eb; border-radius: 8px; color: var(--muted); font-size: 11px; text-align: center; }
    .inline-note { margin-top: 14px; padding: 11px 13px; border: 1px solid #e7ebf3; border-radius: 8px; background: #f8f9fc; color: #66748a; font-size: 11px; line-height: 1.6; }
    .tag { display: inline-flex; align-items: center; gap: 6px; min-height: 23px; padding: 0 8px; border-radius: 999px; background: #f0f3f8; color: #526078; font-size: 10px; font-weight: 650; white-space: nowrap; }
    .tag.blue { background: var(--blue-soft); color: #4059ca; }
    .tag.green { background: var(--green-soft); color: #187454; }
    .tag.amber { background: var(--amber-soft); color: #9c5a0e; }
    .tag.red { background: var(--red-soft); color: #ad3948; }
    .tag .service-dot { width: 6px; height: 6px; flex-basis: 6px; box-shadow: none; }
    .table-tools { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
    .table-search { width: min(330px, 100%); display: flex; align-items: center; gap: 8px; padding: 9px 11px; border: 1px solid var(--line); border-radius: 8px; background: white; }
    .table-search svg { width: 15px; height: 15px; color: var(--muted); }
    .table-search input { width: 100%; border: 0; outline: 0; color: var(--ink); font-size: 12px; }
    .table-meta { color: var(--muted); font-size: 11px; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; white-space: nowrap; }
    th { padding: 11px 13px; border-bottom: 1px solid var(--line); background: #fafbfd; color: #7c899c; font-size: 10px; font-weight: 720; letter-spacing: .06em; text-align: left; text-transform: uppercase; }
    th button { padding: 0; border: 0; background: transparent; color: inherit; font: inherit; letter-spacing: inherit; text-transform: inherit; }
    td { padding: 12px 13px; border-bottom: 1px solid #eff2f6; color: #46546a; font-size: 11px; }
    tr:last-child td { border-bottom: 0; }
    tbody tr:hover { background: #fafbfe; }
    .customer-id { color: var(--ink); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; font-weight: 650; }
    .table-action { border: 0; background: transparent; color: var(--blue); font-size: 11px; font-weight: 650; }
    .table-action:hover { text-decoration: underline; }
    .pagination { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 13px; border-top: 1px solid var(--line); color: var(--muted); font-size: 11px; }
    .pagination-controls { display: flex; gap: 6px; }
    .pagination .button { min-height: 30px; padding: 0 9px; font-size: 11px; }
    .selection-panel { display: grid; grid-template-columns: minmax(0,1fr) auto; align-items: end; gap: 12px; padding: 17px; }
    .field-label { display: block; margin-bottom: 7px; color: #56647a; font-size: 11px; font-weight: 650; }
    .field-input { width: 100%; min-height: 39px; padding: 0 11px; border: 1px solid #dfe5ed; border-radius: 8px; background: white; color: var(--ink); }
    .result-grid { grid-template-columns: repeat(3, minmax(0,1fr)); margin-top: 16px; }
    .result-card { padding: 17px; }
    .result-label { color: var(--muted); font-size: 10px; font-weight: 680; letter-spacing: .07em; text-transform: uppercase; }
    .result-value { margin-top: 9px; color: var(--ink); font-size: 23px; font-weight: 740; letter-spacing: -.03em; }
    .result-detail { margin-top: 6px; color: var(--muted); font-size: 11px; line-height: 1.55; }
    .probability-track { height: 10px; margin-top: 12px; overflow: hidden; border-radius: 99px; background: #eef1f6; }
    .probability-track span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #5776ed, #745be0); transition: width .4s; }
    .section-gap { margin-top: 17px; }
    .model-strip { display: flex; flex-wrap: wrap; gap: 9px; }
    .model-chip { flex: 1 1 130px; padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfe; }
    .model-chip span { display: block; color: var(--muted); font-size: 10px; }
    .model-chip strong { display: block; margin-top: 5px; color: var(--ink); font-size: 15px; }
    .explanation { display: flex; gap: 11px; padding: 14px; border: 1px solid #e2e8fb; border-radius: 9px; background: #f7f8ff; color: #53627c; font-size: 11px; line-height: 1.65; }
    .explanation svg { width: 17px; height: 17px; flex: 0 0 17px; color: var(--blue); }
    .feature-list { display: grid; gap: 9px; }
    .feature-row { display: grid; grid-template-columns: 1fr 120px; gap: 12px; padding-bottom: 9px; border-bottom: 1px solid #eef1f5; color: var(--muted); font-size: 11px; }
    .feature-row strong { color: var(--ink); text-align: right; font-weight: 650; font-variant-numeric: tabular-nums; }
    .priority-meter { position: relative; height: 13px; margin: 19px 0 10px; border-radius: 20px; background: linear-gradient(90deg, #53b68d 0 50%, #e9b34f 50% 75%, #d76b75 75%); }
    .priority-pin { position: absolute; top: -5px; width: 3px; height: 23px; border: 1px solid white; border-radius: 3px; background: #17243a; box-shadow: 0 0 0 1px #17243a; transform: translateX(-50%); }
    .meter-labels { display: flex; justify-content: space-between; color: var(--faint); font-size: 9px; }
    .action-list { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 10px; }
    .action-card { padding: 14px; border: 1px solid var(--line); border-radius: 9px; background: white; transition: border-color .15s, background .15s; }
    .action-card.selected { border-color: #8798f1; background: #f5f6ff; box-shadow: 0 0 0 2px rgba(82,105,227,.08); }
    .action-name { color: var(--ink); font-size: 12px; font-weight: 690; }
    .action-copy { margin-top: 6px; color: var(--muted); font-size: 10px; line-height: 1.55; }
    .action-card .tag { margin-top: 10px; }
    .workflow { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 11px; }
    .workflow-step { position: relative; min-height: 126px; padding: 15px; border: 1px solid var(--line); border-radius: 9px; background: white; }
    .workflow-number { width: 24px; height: 24px; display: grid; place-items: center; border-radius: 7px; background: var(--blue-soft); color: var(--blue); font-size: 10px; font-weight: 750; }
    .workflow-step h3 { margin: 12px 0 5px; font-size: 12px; }
    .workflow-step p { margin: 0; color: var(--muted); font-size: 10px; line-height: 1.55; }
    .alert { margin: 12px 0; padding: 11px 13px; border: 1px solid #f0d4d7; border-radius: 8px; background: var(--red-soft); color: #9e3d49; font-size: 11px; line-height: 1.55; }
    .alert[hidden] { display: none; }
    .overlay { display: none; }
    .skeleton { height: 13px; border-radius: 6px; background: linear-gradient(90deg,#eff2f6,#f8f9fb,#eff2f6); background-size: 200% 100%; animation: shimmer 1.2s linear infinite; }
    @keyframes shimmer { to { background-position: -200% 0; } }
    .spinner { width: 14px; height: 14px; border: 2px solid rgba(255,255,255,.45); border-top-color: white; border-radius: 50%; animation: spin .7s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @media (max-width: 1120px) {
      .sidebar { width: 224px; }
      .main { margin-left: 224px; }
      .topbar { padding: 0 24px; }
      .content { padding: 26px 24px 42px; }
      .kpi-grid { grid-template-columns: repeat(2,minmax(0,1fr)); }
      .workflow { grid-template-columns: repeat(2,minmax(0,1fr)); }
    }
    @media (max-width: 760px) {
      .sidebar { width: 264px; transform: translateX(-100%); box-shadow: 12px 0 32px rgba(14,25,43,.18); }
      body.nav-open .sidebar { transform: translateX(0); }
      .main { margin-left: 0; }
      .overlay { position: fixed; z-index: 25; inset: 0; background: rgba(9,18,32,.42); }
      body.nav-open .overlay { display: block; }
      .topbar { height: 61px; padding: 0 15px; }
      .menu-toggle { display: inline-flex; }
      .breadcrumb { overflow: hidden; text-overflow: ellipsis; }
      .lookup { width: 38px; padding: 9px; cursor: pointer; }
      .lookup input { display: none; }
      .lookup:focus-within { position: absolute; left: 58px; right: 98px; width: auto; background: white; }
      .lookup:focus-within input { display: block; }
      .refresh-label, .avatar { display: none; }
      .content { padding: 23px 15px 36px; }
      .page-heading { align-items: flex-start; flex-direction: column; margin-bottom: 18px; }
      .kpi-grid { gap: 10px; }
      .kpi { min-height: 116px; padding: 14px; }
      .kpi-value { font-size: 23px; }
      .two-col, .three-col, .result-grid { grid-template-columns: 1fr; }
      .selection-panel { grid-template-columns: 1fr; }
      .selection-panel .button { width: 100%; }
      .workflow { grid-template-columns: repeat(2,minmax(0,1fr)); }
      .action-list { grid-template-columns: 1fr; }
      .table-tools { align-items: flex-start; flex-direction: column; }
    }
    @media (max-width: 390px) {
      .kpi-grid { grid-template-columns: 1fr 1fr; }
      .kpi-label { align-items: flex-start; font-size: 10px; }
      .kpi-icon { width: 25px; height: 25px; }
      .workflow { grid-template-columns: 1fr; }
      .top-actions { gap: 5px; }
    }
  </style>
</head>
<body>
<svg aria-hidden="true" style="position:absolute;width:0;height:0;overflow:hidden" xmlns="http://www.w3.org/2000/svg">
  <symbol id="i-grid" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></symbol>
  <symbol id="i-users" viewBox="0 0 24 24"><path d="M16 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="10" cy="7" r="4"/><path d="M20 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/></symbol>
  <symbol id="i-pulse" viewBox="0 0 24 24"><path d="M3 12h4l3-8 4 16 3-8h4"/></symbol>
  <symbol id="i-activity" viewBox="0 0 24 24"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></symbol>
  <symbol id="i-target" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></symbol>
  <symbol id="i-spark" viewBox="0 0 24 24"><path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3Z"/><path d="m19 14 1 2.5 2.5 1-2.5 1L19 21l-1-2.5-2.5-1 2.5-1L19 14Z"/></symbol>
  <symbol id="i-book" viewBox="0 0 24 24"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z"/></symbol>
  <symbol id="i-search" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></symbol>
  <symbol id="i-refresh" viewBox="0 0 24 24"><path d="M20 7v5h-5M4 17v-5h5"/><path d="M5.6 9A7 7 0 0 1 17 6l3 6M4 12l3 6a7 7 0 0 0 11.4-3"/></symbol>
  <symbol id="i-menu" viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h16"/></symbol>
  <symbol id="i-user" viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M5 21v-2a7 7 0 0 1 14 0v2"/></symbol>
  <symbol id="i-box" viewBox="0 0 24 24"><path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 8 9 5 9-5M3 16l9 5 9-5M3 8v8m18-8v8M12 13v8"/></symbol>
  <symbol id="i-clock" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></symbol>
  <symbol id="i-info" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v5m0-8h.01"/></symbol>
</svg>
<div class="app-shell">
  <aside class="sidebar" id="sidebar">
    <div class="brand">
      <div class="brand-row"><div class="brand-mark"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-pulse"/></svg></div><div class="brand-name">Adaptive Hybrid<br>Customer Retention</div></div>
      <div class="brand-caption">Customer intelligence</div>
    </div>
    <div class="nav-section">Workspace</div>
    <nav class="nav-links" aria-label="Main navigation">
      <button class="nav-link active" data-page="overview"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-grid"/></svg>Overview</button>
      <button class="nav-link" data-page="customers"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-users"/></svg>Customer Analysis</button>
      <button class="nav-link" data-page="churn"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-pulse"/></svg>Churn Prediction</button>
      <button class="nav-link" data-page="drift"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-activity"/></svg>Behaviour Drift</button>
      <button class="nav-link" data-page="priority"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-target"/></svg>Retention Priority (CRPI)</button>
      <button class="nav-link" data-page="actions"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-spark"/></svg>Recommended Actions</button>
    </nav>
    <div class="nav-section">Project</div>
    <nav class="nav-links" aria-label="Project information">
      <button class="nav-link" data-page="about"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-book"/></svg>About the Project</button>
    </nav>
    <div class="sidebar-bottom">
      <div class="service-card"><div class="service-top"><span class="service-dot" id="service-dot"></span><span id="service-label">Checking inference assets</span></div><div class="service-note" id="service-note">Verifying required model and observation files.</div></div>
      <div class="sidebar-footer">Adaptive Hybrid Customer Retention · Research demo</div>
    </div>
  </aside>
  <div class="overlay" id="nav-overlay"></div>
  <main class="main">
    <header class="topbar">
      <div class="top-left">
        <button class="icon-button menu-toggle" id="menu-toggle" aria-label="Open navigation"><svg fill="none" stroke="currentColor" stroke-width="1.8"><use href="#i-menu"/></svg></button>
        <div class="breadcrumb"><span>Customer intelligence</span> &nbsp;/&nbsp; <strong id="breadcrumb-title">Overview</strong></div>
      </div>
      <div class="top-actions">
        <form class="lookup" id="global-search-form" role="search"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-search"/></svg><input id="global-search" autocomplete="off" placeholder="Find customer ID…" aria-label="Find customer ID"></form>
        <button class="icon-button" id="refresh-button" aria-label="Refresh current page" title="Refresh current page"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-refresh"/></svg></button>
        <div class="avatar" aria-label="Project demonstration">AH</div>
      </div>
    </header>
    <div class="content">
      <section class="page active" id="page-overview">
        <div class="page-heading"><div><div class="eyebrow">System overview</div><h1>Customer retention overview</h1><p class="page-description">A live view of the observation dataset and analyses run in this browser session. Model-derived distributions appear as customers are analysed.</p></div><div class="heading-actions"><button class="button" data-goto="customers">Browse customers</button><button class="button primary" data-goto="churn">Run an analysis</button></div></div>
        <div class="grid kpi-grid">
          <article class="card kpi"><div class="kpi-label">Customers available<span class="kpi-icon"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-users"/></svg></span></div><div class="kpi-value" id="kpi-customers">—</div><div class="kpi-foot">Observation-period feature records</div></article>
          <article class="card kpi"><div class="kpi-label">Analysed in this session<span class="kpi-icon violet"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-pulse"/></svg></span></div><div class="kpi-value" id="kpi-analysed">0</div><div class="kpi-foot">Unique inference results in this browser tab</div></article>
          <article class="card kpi"><div class="kpi-label">High churn risk<span class="kpi-icon amber"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-target"/></svg></span></div><div class="kpi-value" id="kpi-high-risk">0</div><div class="kpi-foot">Probability at or above 50% threshold</div></article>
          <article class="card kpi"><div class="kpi-label">Behaviour drift detected<span class="kpi-icon green"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-activity"/></svg></span></div><div class="kpi-value" id="kpi-drift">0</div><div class="kpi-foot">Above the configured drift threshold</div></article>
        </div>
        <div class="grid two-col">
          <article class="card"><div class="card-head"><div><h2 class="card-title">Churn probability distribution</h2><p class="card-subtitle">Actual model results from analyses in this session</p></div><span class="tag blue" id="chart-analysis-count">0 analysed</span></div><div class="card-body"><div class="chart" id="chart-churn"></div></div></article>
          <article class="card"><div class="card-head"><div><h2 class="card-title">Retention priority distribution</h2><p class="card-subtitle">CRPI categories from actual inference results</p></div></div><div class="card-body"><div class="chart" id="chart-priority"></div></div></article>
          <article class="card"><div class="card-head"><div><h2 class="card-title">Drift severity</h2><p class="card-subtitle">Euclidean distance between early and later GRU hidden-state means</p></div></div><div class="card-body"><div class="chart" id="chart-drift"></div></div></article>
          <article class="card"><div class="card-head"><div><h2 class="card-title">LinUCB action distribution</h2><p class="card-subtitle">Selected trained-policy action; no simulated campaign outcomes</p></div></div><div class="card-body"><div class="chart" id="chart-actions"></div></div></article>
        </div>
        <div class="inline-note" id="dataset-note">Loading observation dataset summary…</div>
      </section>

      <section class="page" id="page-customers">
        <div class="page-heading"><div><div class="eyebrow">Observation dataset</div><h1>Customer analysis</h1><p class="page-description">Search and inspect real customer-level observation features. Customer identifiers are pseudonymous; no direct personal details are shown.</p></div></div>
        <article class="card">
          <div class="card-head"><div><h2 class="card-title">Customer records</h2><p class="card-subtitle">Paginated records from the prepared observation-period feature dataset</p></div><span class="tag" id="customer-count-tag">Loading</span></div>
          <div class="card-body">
            <div class="table-tools"><form class="table-search" id="customer-search-form"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-search"/></svg><input id="customer-search" placeholder="Search customer ID" aria-label="Search customers"></form><div class="table-meta" id="table-meta">Loading customer records…</div></div>
            <div class="alert" id="customer-error" hidden></div>
            <div class="table-wrap"><table><thead><tr>
              <th><button data-sort="customer_unique_id">Customer ID</button></th><th><button data-sort="total_orders">Orders</button></th><th>Observed orders</th><th><button data-sort="total_spending">Historical spend</button></th><th><button data-sort="avg_order_value">Avg. order value</button></th><th><button data-sort="recency_days">Recency</button></th><th><button data-sort="avg_review_score">Avg. review</button></th><th>Actions</th>
            </tr></thead><tbody id="customer-rows"><tr><td colspan="8">Loading…</td></tr></tbody></table></div>
            <div class="pagination"><span id="pagination-label">—</span><div class="pagination-controls"><button class="button" id="page-prev">Previous</button><button class="button" id="page-next">Next</button></div></div>
          </div>
        </article>
        <article class="card section-gap" id="customer-detail-card" hidden><div class="card-head"><div><h2 class="card-title">Customer details</h2><p class="card-subtitle" id="customer-detail-id">Select a customer to inspect the available features.</p></div><button class="button" id="detail-analyze">Run inference</button></div><div class="card-body"><div class="feature-list" id="customer-detail-fields"></div></div></article>
      </section>

      <section class="page" id="page-churn">
        <div class="page-heading"><div><div class="eyebrow">Trained inference</div><h1>Churn prediction</h1><p class="page-description">Run the saved Random Forest, XGBoost, GRU sequence encoder and logistic-regression fusion pipeline for a real customer or built-in synthetic demo profile.</p></div><span class="tag" id="inference-mode-tag">Awaiting analysis</span></div>
        <article class="card"><div class="selection-panel"><div><label class="field-label" for="analysis-customer-id">Customer identifier</label><input class="field-input" id="analysis-customer-id" placeholder="Paste an observation customer ID or choose a demo profile"></div><div><label class="field-label" for="demo-select">Demo profile</label><select class="field-input" id="demo-select"><option value="">Choose profile</option></select></div><button class="button primary" id="analyze-button">Run trained inference</button></div></article>
        <div class="alert" id="analysis-error" hidden></div>
        <div class="grid result-grid">
          <article class="card result-card"><div class="result-label">Final churn probability</div><div class="result-value" id="result-probability">—</div><span class="tag" id="result-risk-tag">No result</span><div class="probability-track"><span id="probability-fill" style="width:0"></span></div><div class="result-detail">High risk is the classifier's documented ≥50% probability threshold.</div></article>
          <article class="card result-card"><div class="result-label">Prediction context</div><div class="result-value" id="result-customer">—</div><div class="result-detail" id="result-explanation">Run inference to see customer-specific model results.</div></article>
          <article class="card result-card"><div class="result-label">Model output</div><div class="model-strip" style="margin-top:10px"><div class="model-chip"><span>Random Forest</span><strong id="result-rf">—</strong></div><div class="model-chip"><span>XGBoost</span><strong id="result-xgb">—</strong></div><div class="model-chip"><span>GRU sequence embedding</span><strong id="result-embedding">—</strong></div></div><div class="result-detail">The two probability outputs and 64 GRU embedding values form the 66-feature fusion input.</div></article>
        </div>
      </section>

      <section class="page" id="page-drift">
        <div class="page-heading"><div><div class="eyebrow">Sequence behaviour</div><h1>Behaviour drift</h1><p class="page-description">Compare recent and earlier customer behaviour representations. Drift is a change signal—not a churn label.</p></div></div>
        <div class="grid two-col">
          <article class="card result-card"><div class="result-label">Selected customer drift severity</div><div class="result-value" id="drift-value">—</div><span class="tag" id="drift-tag">Awaiting analysis</span><div class="result-detail" id="drift-detail">Run inference to calculate a customer-specific drift score.</div></article>
          <article class="card result-card"><div class="result-label">Page-Hinkley detector</div><div class="result-value" id="page-hinkley-value">—</div><div class="result-detail" id="page-hinkley-detail">Detector status is returned by the inference pipeline.</div></article>
        </div>
        <article class="card section-gap"><div class="card-head"><div><h2 class="card-title">How to interpret this signal</h2><p class="card-subtitle">Method used by the current trained pipeline</p></div></div><div class="card-body"><div class="explanation"><svg fill="none" stroke="currentColor" stroke-width="1.7"><use href="#i-info"/></svg><div>Drift severity is the Euclidean distance between the mean GRU hidden-state representations of earlier and later portions of the customer's order sequence. The configured threshold determines the drift flag. A drift alert indicates behaviour changed; it does not, by itself, mean the customer has churned.</div></div></div></article>
      </section>

      <section class="page" id="page-priority">
        <div class="page-heading"><div><div class="eyebrow">Customer prioritisation</div><h1>Retention Priority Index</h1><p class="page-description">The existing CRPI formula combines normalized churn probability, historical CLV proxy and observed behaviour drift using the saved scalers.</p></div></div>
        <div class="grid two-col">
          <article class="card result-card"><div class="result-label">Current CRPI</div><div class="result-value" id="crpi-value">—</div><span class="tag" id="crpi-priority">Awaiting analysis</span><div class="priority-meter"><span class="priority-pin" id="crpi-pin" style="left:0"></span></div><div class="meter-labels"><span>Lower priority</span><span>Higher priority</span></div></article>
          <article class="card result-card"><div class="result-label">Historical customer lifetime value proxy</div><div class="result-value" id="clv-value">—</div><div class="result-detail">Derived from observed historical spend and customer lifetime features. This is a historical proxy, not a guaranteed prediction of future revenue.</div></article>
        </div>
        <article class="card section-gap"><div class="card-head"><div><h2 class="card-title">CRPI components</h2><p class="card-subtitle">Values returned by the unchanged inference pipeline</p></div></div><div class="card-body"><div class="feature-list"><div class="feature-row"><span>Final churn probability</span><strong id="crpi-churn-component">—</strong></div><div class="feature-row"><span>Normalized historical CLV proxy</span><strong id="crpi-clv-component">—</strong></div><div class="feature-row"><span>Behaviour drift severity</span><strong id="crpi-drift-component">—</strong></div></div><div class="inline-note">The existing CRPI formula and fitted normalization scalers are used as-is. Priority categories use the thresholds stored with the trained artifacts.</div></div></article>
      </section>

      <section class="page" id="page-actions">
        <div class="page-heading"><div><div class="eyebrow">Policy recommendation</div><h1>Recommended actions</h1><p class="page-description">The selected response is the action returned by the trained LinUCB policy for the analyzed customer context.</p></div></div>
        <article class="card result-card"><div class="result-label">LinUCB selected action</div><div class="result-value" id="selected-action">—</div><div class="result-detail" id="action-context">Run inference to see the action selected by the current policy.</div></article>
        <article class="card section-gap"><div class="card-head"><div><h2 class="card-title">Available retention actions</h2><p class="card-subtitle">The action selected by inference is highlighted</p></div></div><div class="card-body"><div class="action-list" id="action-cards"></div></div></article>
        <div class="inline-note">No manual demonstration override is active in this interface. The action shown is the trained LinUCB policy output. Offline or synthetic reward simulations are not real campaign outcomes.</div>
      </section>

      <section class="page" id="page-about">
        <div class="page-heading"><div><div class="eyebrow">Engineering project</div><h1>About the system</h1><p class="page-description">An interpretable customer-retention workflow combining static behaviour, sequential representation learning, drift monitoring and contextual action selection.</p></div></div>
        <article class="card"><div class="card-head"><div><h2 class="card-title">End-to-end workflow</h2><p class="card-subtitle">Existing trained artifacts and inference logic, presented as a human-readable pipeline</p></div></div><div class="card-body"><div class="workflow">
          <div class="workflow-step"><span class="workflow-number">01</span><h3>Olist dataset</h3><p>Prepared customer and order records provide the historical observation window.</p></div>
          <div class="workflow-step"><span class="workflow-number">02</span><h3>Preprocessing</h3><p>Order events are cleaned, organized and aligned to the model's expected inputs.</p></div>
          <div class="workflow-step"><span class="workflow-number">03</span><h3>Customer features</h3><p>Static purchasing, value, review and recency features summarize customer history.</p></div>
          <div class="workflow-step"><span class="workflow-number">04</span><h3>RF + XGBoost + GRU</h3><p>Tree models estimate churn from static features; GRU represents order sequences.</p></div>
          <div class="workflow-step"><span class="workflow-number">05</span><h3>66-feature fusion</h3><p>Two base-model probabilities and 64 GRU embedding values are combined.</p></div>
          <div class="workflow-step"><span class="workflow-number">06</span><h3>Logistic regression</h3><p>The saved fusion model produces the final churn probability.</p></div>
          <div class="workflow-step"><span class="workflow-number">07</span><h3>Drift detection + CRPI</h3><p>Hidden-state change is measured and combined with churn and historical value signals.</p></div>
          <div class="workflow-step"><span class="workflow-number">08</span><h3>LinUCB action</h3><p>The contextual policy selects one of the configured retention actions.</p></div>
        </div></div></article>
        <div class="grid two-col section-gap"><article class="card result-card"><div class="result-label">Inference assets</div><div class="result-value" style="font-size:17px" id="about-model-status">Checking…</div><div class="result-detail" id="about-model-note">Status reflects file availability, not a guarantee that every prediction will succeed.</div></article><article class="card result-card"><div class="result-label">Evaluation boundaries</div><div class="result-detail" style="margin-top:12px">The interface reports observed features and actual inference outputs. It does not claim campaign lift, guaranteed future revenue, or model performance scores unless a verified source for those metrics is available.</div></article></div>
      </section>
    </div>
  </main>
</div>
<script>
(() => {
  'use strict';
  const titles = {overview:'Overview', customers:'Customer Analysis', churn:'Churn Prediction', drift:'Behaviour Drift', priority:'Retention Priority (CRPI)', actions:'Recommended Actions', about:'About the Project'};
  const actionDescriptions = {
    'No Intervention':'No retention contact is recommended for this context.',
    'Personalized Recommendation':'Use relevant product recommendations to support engagement.',
    'Loyalty Incentive':'Consider a loyalty-oriented incentive for this customer.',
    'Proactive Retention Offer':'Prioritize a proactive retention contact or offer.'
  };
  const numberFormat = new Intl.NumberFormat(undefined, {maximumFractionDigits:0});
  const moneyFormat = new Intl.NumberFormat(undefined, {style:'currency', currency:'INR', maximumFractionDigits:0});
  const state = {
    page:'overview', customerPage:1, pageSize:20, query:'', sortBy:'total_spending', sortDirection:'desc',
    totalPages:1, totalCustomers:0, selectedCustomer:null, result:null, analyses:[], demos:[]
  };
  try {
    state.analyses = JSON.parse(sessionStorage.getItem('retention-analysis-session') || '[]');
    if (!Array.isArray(state.analyses)) state.analyses = [];
  } catch (_) { state.analyses = []; }

  const el = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const fmtPct = (n) => Number.isFinite(Number(n)) ? `${(Number(n)*100).toFixed(1)}%` : '—';
  const fmtNum = (n, places=1) => Number.isFinite(Number(n)) ? Number(n).toLocaleString(undefined,{maximumFractionDigits:places}) : '—';
  const fmtMoney = (n) => Number.isFinite(Number(n)) ? moneyFormat.format(Number(n)) : '—';

  async function api(path, options={}) {
    const response = await fetch(path, options);
    let body;
    try { body = await response.json(); } catch (_) { throw new Error(`Server returned an unreadable response (${response.status}).`); }
    if (!response.ok) throw new Error(body.error || `Request failed (${response.status}).`);
    return body;
  }

  function setPage(page) {
    if (!titles[page]) return;
    state.page = page;
    document.querySelectorAll('.page').forEach(section => section.classList.toggle('active', section.id === `page-${page}`));
    document.querySelectorAll('.nav-link').forEach(link => link.classList.toggle('active', link.dataset.page === page));
    el('breadcrumb-title').textContent = titles[page];
    document.body.classList.remove('nav-open');
    if (page === 'customers' && !el('customer-rows').dataset.loaded) loadCustomers();
    if (page === 'overview') renderOverviewCharts();
    if (page === 'drift') renderDriftPage();
    if (page === 'priority') renderPriorityPage();
    if (page === 'actions') renderActionPage();
  }

  async function loadOverview() {
    try {
      const data = await api('/api/overview');
      state.totalCustomers = data.customer_count;
      el('kpi-customers').textContent = numberFormat.format(data.customer_count);
      el('dataset-note').textContent = `${numberFormat.format(data.customer_count)} unique customer records and ${numberFormat.format(data.observation_order_count)} observation-period orders. Dataset averages: ${fmtMoney(data.average_spend)} historical spend per customer, ${fmtMoney(data.average_order_value)} average order value, and ${fmtNum(data.average_review_score,2)} average review score.`;
    } catch (error) {
      el('kpi-customers').textContent = '—';
      el('dataset-note').textContent = `Dataset summary unavailable: ${error.message}`;
    }
    renderOverviewCharts();
  }

  function barChart(targetId, items, color='', axisLabel='Analysed customers') {
    const target = el(targetId);
    const total = items.reduce((sum, item) => sum + item.value, 0);
    if (!items.length || total === 0) {
      target.innerHTML = '<div class="empty-chart">Run one or more customer analyses to populate this model-derived chart.</div>';
      return;
    }
    const max = Math.max(1, ...items.map(item => item.value));
    target.innerHTML = `<div class="bar-list">${items.map(item => `<div class="bar-row"><span>${esc(item.label)}</span><div class="bar-track" title="${esc(`${item.label}: ${fmtNum(item.value,item.decimals||0)} ${item.unit||'customers'}`)}"><div class="bar-fill ${esc(item.color || color)}" style="width:${Math.max(item.value ? 4 : 0, item.value/max*100)}%"></div></div><span class="bar-count">${fmtNum(item.value,item.decimals||0)}</span></div>`).join('')}<div class="chart-axis"><span>${esc(axisLabel)}</span><span>${fmtNum(max,items.some(item=>item.decimals)?3:0)}</span></div></div>`;
  }

  function renderOverviewCharts() {
    const values = state.analyses;
    const churnBins = [
      {label:'0–19%',value:0,color:'green'}, {label:'20–39%',value:0,color:'green'},
      {label:'40–59%',value:0,color:'amber'}, {label:'60–79%',value:0,color:'amber'},
      {label:'80–100%',value:0,color:'red'}
    ];
    values.forEach(item => { const p=Number(item.final_churn_probability); const idx=Math.min(4,Math.floor(Math.max(0,p)*5)); if(Number.isFinite(p)) churnBins[idx].value++; });
    barChart('chart-churn', churnBins);
    const priorityNames=['Low Priority','Medium Priority','High Priority'];
    barChart('chart-priority',priorityNames.map((label,i)=>({label,value:values.filter(x=>String(x.retention_priority).startsWith(['Low','Medium','High'][i])).length,color:['green','amber','red'][i]})));
    const recent=values.slice(-8);
    barChart('chart-drift',recent.map(x=>({label:String(x.customer_id).slice(0,14),value:Math.max(0,Number(x.behaviour_drift_score)||0),decimals:3,unit:'drift score',color:Number(x.high_drift)?'amber':'blue'})),'','Drift severity score');
    const actions=['No Intervention','Personalized Recommendation','Loyalty Incentive','Proactive Retention Offer'];
    barChart('chart-actions',actions.map((label,i)=>({label,value:values.filter(x=>x.base_linucb_action===label).length,color:['green','blue','violet','amber'][i]})));
    el('kpi-analysed').textContent=numberFormat.format(values.length);
    el('kpi-high-risk').textContent=numberFormat.format(values.filter(x=>Number(x.final_churn_probability)>=0.5).length);
    el('kpi-drift').textContent=numberFormat.format(values.filter(x=>Boolean(x.high_drift)).length);
    el('chart-analysis-count').textContent=`${values.length} analysed`;
  }

  async function loadCustomers() {
    const tbody=el('customer-rows');
    el('customer-error').hidden=true;
    tbody.innerHTML='<tr><td colspan="8"><div class="skeleton" style="width:100%"></div></td></tr>';
    const params=new URLSearchParams({query:state.query,page:String(state.customerPage),page_size:String(state.pageSize),sort_by:state.sortBy,sort_direction:state.sortDirection});
    try {
      const data=await api(`/api/customers?${params}`);
      state.totalPages=Math.max(1,data.pages);
      el('customer-rows').dataset.loaded='true';
      el('table-meta').textContent=`${numberFormat.format(data.total)} matching records`;
      el('customer-count-tag').textContent=`${numberFormat.format(data.total)} customers`;
      el('pagination-label').textContent=data.total?`Page ${data.page} of ${Math.max(1,data.pages)}`:'No records';
      el('page-prev').disabled=data.page<=1;
      el('page-next').disabled=data.page>=data.pages;
      if (!data.items.length) {
        tbody.innerHTML='<tr><td colspan="8" style="padding:28px;text-align:center;color:#728096">No matching customers. Try another identifier.</td></tr>';
        return;
      }
      tbody.innerHTML=data.items.map(row=>`<tr>
        <td><button class="table-action customer-id" data-detail="${esc(row.customer_unique_id)}">${esc(row.customer_unique_id)}</button></td>
        <td>${fmtNum(row.total_orders,0)}</td><td>${fmtNum(row.observed_order_count,0)}${row.inference_available?'':' <span class="tag">Not enough for GRU</span>'}</td><td>${fmtMoney(row.total_spending)}</td><td>${fmtMoney(row.avg_order_value)}</td>
        <td>${fmtNum(row.recency_days,0)} days</td><td>${fmtNum(row.avg_review_score,1)} / 5</td>
        <td>${row.inference_available?`<button class="table-action" data-analyze="${esc(row.customer_unique_id)}">Analyze</button>`:'<span class="table-meta">Requires 2+ observed orders</span>'}</td>
      </tr>`).join('');
    } catch(error) {
      tbody.innerHTML='<tr><td colspan="8" style="padding:24px;text-align:center;color:#ad3948">Customer records could not be loaded.</td></tr>';
      el('customer-error').textContent=error.message;
      el('customer-error').hidden=false;
      el('table-meta').textContent='Unavailable';
    }
  }

  async function showCustomerDetail(customerId) {
    try {
      const customer=await api(`/api/customer?customer_id=${encodeURIComponent(customerId)}`);
      state.selectedCustomer=customer;
      el('customer-detail-card').hidden=false;
      el('customer-detail-id').textContent=customer.customer_unique_id;
      el('detail-analyze').disabled=!customer.inference_available;
      el('detail-analyze').textContent=customer.inference_available?'Run inference':'Requires 2+ observed orders';
      const fields=[
        ['Total orders',fmtNum(customer.total_orders,0)],['Total items',fmtNum(customer.total_items,0)],
        ['Orders in observation window',fmtNum(customer.observed_order_count,0)],
        ['Historical spend',fmtMoney(customer.total_spending)],['Average order value',fmtMoney(customer.avg_order_value)],
        ['Average review score',`${fmtNum(customer.avg_review_score,1)} / 5`],['Recency',`${fmtNum(customer.recency_days,0)} days`],
        ['Observed customer lifetime',`${fmtNum(customer.customer_lifetime_days,0)} days`]
      ];
      el('customer-detail-fields').innerHTML=fields.map(([label,value])=>`<div class="feature-row"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join('');
      el('customer-detail-card').scrollIntoView({behavior:'smooth',block:'nearest'});
    } catch(error) { showNotice('customer-error',error.message); }
  }

  function showNotice(id,message) { const node=el(id); node.textContent=message; node.hidden=false; }
  function hideNotice(id) { const node=el(id); node.hidden=true; node.textContent=''; }

  async function startAnalysis(customerId) {
    const id=String(customerId||'').trim();
    if(!id) { showNotice('analysis-error','Enter a customer identifier or choose a demo profile.'); return; }
    hideNotice('analysis-error');
    const button=el('analyze-button');
    button.disabled=true;
    button.innerHTML='<span class="spinner"></span> Analysing…';
    try {
      const payload={customer_id:id};
      const result=await api('/api/analyze',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded;charset=UTF-8'},body:new URLSearchParams(payload)});
      state.result=result;
      state.analyses=state.analyses.filter(item=>item.customer_id!==result.customer_id);
      state.analyses.push(result);
      try { sessionStorage.setItem('retention-analysis-session',JSON.stringify(state.analyses)); } catch (_) {}
      renderResult(result);
      setPage('churn');
    } catch(error) {
      showNotice('analysis-error',error.message);
    } finally {
      button.disabled=false;
      button.textContent='Run trained inference';
    }
  }

  function renderResult(result) {
    el('analysis-customer-id').value=result.customer_id;
    el('demo-select').value=result.customer_id.startsWith('DEMO_')?result.customer_id:'';
    el('result-probability').textContent=fmtPct(result.final_churn_probability);
    const high=Number(result.final_churn_probability)>=0.5;
    const riskTag=el('result-risk-tag');
    riskTag.className=`tag ${high?'red':'green'}`;
    riskTag.textContent=result.prediction_label || (high?'HIGH CHURN RISK':'LOW CHURN RISK');
    el('probability-fill').style.width=`${Math.max(0,Math.min(100,Number(result.final_churn_probability)*100))}%`;
    el('result-customer').textContent=result.display_name || result.customer_id;
    el('result-explanation').textContent=`${result.customer_id} has a ${fmtPct(result.final_churn_probability)} final churn probability. The system classifies this as ${high?'high':'low'} risk using the ≥50% threshold. This is a model estimate, not a certainty.`;
    el('result-rf').textContent=fmtPct(result.rf_probability);
    el('result-xgb').textContent=fmtPct(result.xgb_probability);
    el('result-embedding').textContent=`${result.gru_embedding_dim || (result.gru_embedding||[]).length}-D`;
    el('inference-mode-tag').textContent=result.mode || 'Trained artifact mode';
    el('drift-value').textContent=fmtNum(result.behaviour_drift_score,4);
    const driftTag=el('drift-tag');
    driftTag.className=`tag ${result.high_drift?'amber':'green'}`;
    driftTag.textContent=result.high_drift?'Drift above configured threshold':'No threshold breach';
    el('drift-detail').textContent=`${result.drift_classification || 'Drift score'} · configured drift flag: ${result.high_drift?'detected':'not detected'}. A drift signal alone does not establish churn.`;
    el('page-hinkley-value').textContent=result.page_hinkley_change_detected || 'Unavailable';
    el('page-hinkley-detail').textContent=result.page_hinkley_change_detected==='YES'?'The Page-Hinkley detector reported a change for this sequence.':'The Page-Hinkley detector did not report a change for this sequence.';
    el('crpi-value').textContent=fmtNum(result.crpi_score,4);
    el('crpi-priority').className=`tag ${String(result.retention_priority).startsWith('High')?'red':String(result.retention_priority).startsWith('Medium')?'amber':'green'}`;
    el('crpi-priority').textContent=result.retention_priority || 'Unavailable';
    el('crpi-pin').style.left=`${Math.max(0,Math.min(100,Number(result.crpi_score)*100))}%`;
    el('clv-value').textContent=fmtMoney(result.clv_proxy);
    el('crpi-churn-component').textContent=fmtPct(result.final_churn_probability);
    el('crpi-clv-component').textContent=fmtNum(result.clv_normalized,4);
    el('crpi-drift-component').textContent=fmtNum(result.behaviour_drift_score,4);
    el('selected-action').textContent=result.base_linucb_action || result.final_retention_action || 'Unavailable';
    el('action-context').textContent=`Selected for ${result.customer_id} from its current churn, CRPI, drift and historical-value context. No campaign result is implied.`;
    renderActionPage();
    renderOverviewCharts();
  }

  function renderDriftPage() { if(state.result) return; }
  function renderPriorityPage() { if(state.result) return; }
  function renderActionPage() {
    const actions=['No Intervention','Personalized Recommendation','Loyalty Incentive','Proactive Retention Offer'];
    const selected=state.result?.base_linucb_action;
    el('action-cards').innerHTML=actions.map((action,index)=>`<div class="action-card ${action===selected?'selected':''}"><div class="action-name">${esc(action)}</div><div class="action-copy">${esc(actionDescriptions[action])}</div>${action===selected?'<span class="tag blue">Selected by LinUCB</span>':''}</div>`).join('');
  }

  async function checkService() {
    try {
      const data=await api('/api/status');
      const dot=el('service-dot');
      dot.className=`service-dot ${data.available?'ready':'error'}`;
      el('service-label').textContent=data.available?'Inference assets detected':'Inference assets missing';
      el('service-note').textContent=data.available?'Required model artifacts and observation data files are present.':'Some required model artifacts or data files are missing.';
      el('about-model-status').textContent=data.available?'Files detected':'Files missing';
      el('about-model-note').textContent=data.available?'Required model artifact and observation data files are present; actual predictions are validated when requested.':`Unavailable components: ${(data.missing_artifacts||[]).join(', ') || 'observation data'}.`;
    } catch(error) {
      el('service-dot').className='service-dot error';
      el('service-label').textContent='Model status unavailable';
      el('service-note').textContent='Could not verify required model assets.';
      el('about-model-status').textContent='Status check failed';
      el('about-model-note').textContent=error.message;
    }
  }

  document.querySelectorAll('.nav-link').forEach(link=>link.addEventListener('click',()=>setPage(link.dataset.page)));
  document.querySelectorAll('[data-goto]').forEach(button=>button.addEventListener('click',()=>setPage(button.dataset.goto)));
  el('menu-toggle').addEventListener('click',()=>document.body.classList.toggle('nav-open'));
  el('nav-overlay').addEventListener('click',()=>document.body.classList.remove('nav-open'));
  el('refresh-button').addEventListener('click',()=>{ if(state.page==='customers') loadCustomers(); else if(state.page==='overview') loadOverview(); else if(state.page==='churn'&&state.result) startAnalysis(state.result.customer_id); else checkService(); });
  el('global-search-form').addEventListener('submit',async event=>{
    event.preventDefault();
    const id=el('global-search').value.trim();
    if(!id) return;
    el('global-search').setCustomValidity('');
    try { if(!id.toUpperCase().startsWith('DEMO_')) await api(`/api/customer?customer_id=${encodeURIComponent(id)}`); el('analysis-customer-id').value=id; setPage('churn'); await startAnalysis(id); }
    catch(error) { el('global-search').setCustomValidity(error.message); el('global-search').reportValidity(); }
  });
  el('global-search').addEventListener('input',()=>el('global-search').setCustomValidity(''));
  el('customer-search-form').addEventListener('submit',event=>{event.preventDefault();state.query=el('customer-search').value.trim();state.customerPage=1;loadCustomers();});
  el('customer-search').addEventListener('input',()=>{if(!el('customer-search').value.trim()){state.query='';state.customerPage=1;loadCustomers();}});
  el('page-prev').addEventListener('click',()=>{if(state.customerPage>1){state.customerPage--;loadCustomers();}});
  el('page-next').addEventListener('click',()=>{if(state.customerPage<state.totalPages){state.customerPage++;loadCustomers();}});
  document.querySelectorAll('[data-sort]').forEach(button=>button.addEventListener('click',()=>{const key=button.dataset.sort;state.sortDirection=state.sortBy===key&&state.sortDirection==='asc'?'desc':'asc';state.sortBy=key;state.customerPage=1;loadCustomers();}));
  el('customer-rows').addEventListener('click',event=>{
    const analyze=event.target.closest('[data-analyze]');
    const detail=event.target.closest('[data-detail]');
    if(analyze){el('analysis-customer-id').value=analyze.dataset.analyze;setPage('churn');startAnalysis(analyze.dataset.analyze);}
    else if(detail) showCustomerDetail(detail.dataset.detail);
  });
  el('detail-analyze').addEventListener('click',()=>{if(state.selectedCustomer?.inference_available){el('analysis-customer-id').value=state.selectedCustomer.customer_unique_id;setPage('churn');startAnalysis(state.selectedCustomer.customer_unique_id);}else if(state.selectedCustomer){showNotice('customer-error','This customer has fewer than two orders in the observation window, so sequence inference is unavailable.');}});
  el('analyze-button').addEventListener('click',()=>startAnalysis(el('analysis-customer-id').value));
  el('analysis-customer-id').addEventListener('keydown',event=>{if(event.key==='Enter')startAnalysis(el('analysis-customer-id').value);});
  el('demo-select').addEventListener('change',()=>{if(el('demo-select').value){el('analysis-customer-id').value=el('demo-select').value;hideNotice('analysis-error');}});

  async function init() {
    await Promise.all([loadOverview(),checkService()]);
    try {
      state.demos=await api('/api/demo-customers');
      el('demo-select').innerHTML='<option value="">Choose profile</option>'+state.demos.map(customer=>`<option value="${esc(customer.customer_id)}">${esc(customer.display_name || customer.customer_id)} · ${esc(customer.risk_profile || 'Demo')}</option>`).join('');
    } catch(error) {
      el('demo-select').innerHTML='<option value="">Demo profiles unavailable</option>';
      el('demo-select').disabled=true;
    }
    renderActionPage();
  }
  init();
})();
</script>
</body>
</html>"""


def send_json(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    data = json.dumps(payload, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _customer_record(customer_id: str) -> dict[str, Any]:
    return get_observation_customer(customer_id)


class DemoHandler(BaseHTTPRequestHandler):
    def _query_value(self, query: dict[str, list[str]], name: str, default: str) -> str:
        return query.get(name, [default])[0]

    def _handle_get(self, parsed: Any) -> None:
        if parsed.path == "/":
            body = HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
            return

        query = parse_qs(parsed.query, keep_blank_values=True)
        if parsed.path == "/api/status":
            missing = get_missing_artifacts()
            required_data = [
                PROJECT_ROOT / "data" / "processed" / "customer_features_observation.csv",
                PROJECT_ROOT / "data" / "processed" / "observation_orders.csv",
            ]
            missing_data = [path.name for path in required_data if not path.is_file()]
            send_json(
                self,
                {
                    "available": not missing and not missing_data,
                    "missing_artifacts": missing,
                    "missing_data": missing_data,
                },
            )
            return

        if parsed.path == "/api/overview":
            send_json(self, get_dataset_overview())
            return

        if parsed.path == "/api/customers":
            result = search_observation_customers(
                query=self._query_value(query, "query", ""),
                page=int(self._query_value(query, "page", "1")),
                page_size=int(self._query_value(query, "page_size", "20")),
                sort_by=self._query_value(query, "sort_by", "total_spending"),
                sort_direction=self._query_value(query, "sort_direction", "desc"),
            )
            send_json(self, result)
            return

        if parsed.path == "/api/customer":
            customer_id = self._query_value(query, "customer_id", "").strip()
            if not customer_id:
                send_json(self, {"error": "customer_id is required."}, 400)
                return
            send_json(self, _customer_record(customer_id))
            return

        if parsed.path == "/api/demo-customers":
            send_json(self, load_demo_customers())
            return

        send_json(self, {"error": "Not found."}, 404)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            self._handle_get(parsed)
        except (KeyError, ValueError) as exc:
            send_json(self, {"error": str(exc)}, 400)
        except (OSError, json.JSONDecodeError) as exc:
            send_json(self, {"error": f"Unable to load requested application data: {exc}"}, 500)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/analyze":
            send_json(self, {"error": "Not found."}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                send_json(self, {"error": "Request body must be between 1 byte and 64 KB."}, 400)
                return
            raw = self.rfile.read(length)
            payload = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
            cleaned = {key: values[0] if values else "" for key, values in payload.items()}
            analysis = analyze_customer(cleaned)
            send_json(self, analysis)
        except (KeyError, ValueError) as exc:
            send_json(self, {"error": str(exc)}, 400)
        except (OSError, RuntimeError) as exc:
            send_json(self, {"error": f"Inference could not be completed: {exc}"}, 500)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DemoHandler)
    print(f"Adaptive Hybrid Customer Retention dashboard running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
