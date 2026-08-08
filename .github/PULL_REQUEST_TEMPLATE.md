## 变更描述

简要描述本次 PR 的变更内容。

## 变更类型

- [ ] 新功能 (feature)
- [ ] Bug 修复 (bugfix)
- [ ] 重构 (refactor)
- [ ] 性能优化 (performance)
- [ ] 测试 (test)
- [ ] 文档 (docs)
- [ ] 其他 (other)

## 测试

- [ ] `cargo fmt --all -- --check` 通过
- [ ] `cargo clippy --workspace --all-features -- -D warnings` 通过
- [ ] `cargo test --workspace --all-features` 通过

## 检查清单

- [ ] 新增功能已附带测试
- [ ] 公共 API 有文档注释
- [ ] 无 `unwrap()` / `expect()` 残留（测试代码除外）
- [ ] 无资源泄漏（文件句柄、子进程、网络连接）
