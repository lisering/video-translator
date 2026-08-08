//! Video Translator 桌面应用入口
//!
//! 基于 Tauri v2 构建，提供跨平台桌面 UI。
//! 前端使用 Next.js + TypeScript + Tailwind CSS。
//!
//! 实际的 Tauri 应用设置在 [`vt_ui::run`] 中完成，
//! `main.rs` 仅作为二进制入口点。

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    vt_ui::run();
}
