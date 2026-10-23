# 远程 Codex 访问本机 ChatGPT 网页：恢复与排障手册

版本：2026-10-18。适用于“Codex 运行在远程 Linux 服务器，ChatGPT 登录态保存在 Windows 本机 Edge”这一部署方式。

本次修订：远程端只使用 Codex IDE 扩展也能配置 MCP，不要求终端存在 `codex` 命令；SSH 隧道优先复用已经能登录的连接配置，`-F none` 仅用于已明确全部连接参数的情况。用户提供的日志确认了占位符未替换和复制 PowerShell 提示符两类命令错误；后续验证进度记录如下。

2026-10-18 对当前远程环境的一次只读检查：`edge-local` 配置表已存在、URL 与本文一致且未显式禁用；`ss -ltn '( sport = :8931 )'` 未显示监听。该时点需要排查的是隧道，而非重复注册 MCP。这是当时的本地观测，不代表后续连接状态。

后续检查发现服务器 `127.0.0.1:8931` 已监听，TCP 可连接，但 HTTP 请求 5 秒未收到字节；用户随后确认问题来自 VS Code 自动转发。以下第 4.1 节补充了仅排除 8931 的持久设置。修复后的端到端连通性仍待验证。

最新用户验证：排除自动转发并重新建立 SSH 隧道后，服务器请求 `/mcp` 收到 `HTTP/1.1 400 Bad Request`（响应 Date：2026-10-18 08:34:57 GMT）。这证明该时点 HTTP 转发链路可达，符合普通 GET 未进行 MCP 初始化时的响应；不等于 Codex 已加载工具或已控制浏览器。当前旧会话工具目录仍无 `edge-local`，下一步按第 4.7 节重载 Codex 并新建会话验证工具。

## 1. 这套连接实际做了什么

它不是让 ChatGPT 网页直接连接远程 Codex，也不是共享 ChatGPT 登录 Cookie。实际链路是：

```text
远程服务器上的 Codex
        │
        │ Streamable HTTP MCP
        ▼
服务器 127.0.0.1:8931
        │
        │ SSH 反向隧道：ssh -R
        ▼
Windows 127.0.0.1:8931
        │
        │ Playwright MCP
        ▼
独立 Edge 配置目录 → 已手工登录的 chatgpt.com
```

因此：

- ChatGPT 登录、验证码和双因素认证全部在 Windows Edge 中手工完成。
- 远程 Codex 只通过浏览器自动化读取或操作网页。
- Windows MCP 服务和 SSH 隧道停止后，远程服务器不能继续访问该浏览器。
- Playwright 的持久化 profile 保存登录状态，但它本身是敏感凭据目录。

官方参考：

- [Microsoft Playwright MCP](https://github.com/microsoft/playwright-mcp)
- [Codex MCP 配置说明](https://developers.openai.com/codex/mcp)
- [Codex IDE 打开配置文件](https://developers.openai.com/codex/config-basic/)
- [OpenSSH 客户端参数](https://man.openbsd.org/ssh)
- [OpenSSH 客户端配置](https://man.openbsd.org/ssh_config)
- [OpenSSH 服务端转发限制](https://man.openbsd.org/sshd_config)

## 2. 所需组件

### Windows 本机

- Microsoft Edge。
- Node.js 18 或更高版本，能使用 `node`、`npx.cmd`。
- Windows OpenSSH 客户端，能使用 `ssh`。
- 能正常 SSH 登录远程服务器。

检查：

```powershell
node --version
npx.cmd --version
ssh -V
```

### 远程服务器

- 正在使用的 Codex IDE 扩展即可；独立安装 Codex CLI 是可选项。
- `curl`。
- 服务器允许 SSH remote forwarding。通常不需要开放公网端口。

远程终端检查：

```bash
curl --version
```

`codex: command not found` 只说明当前 shell 找不到该命令，不代表 IDE 扩展不能工作。扩展可能自带内部程序，但其目录未必出现在用户终端的 PATH 中；不要把扩展的带版本内部路径写成永久依赖。直接使用第 4.6 节的配置文件方式。

## 3. 约定参数

本文使用：

```text
MCP 名称：edge-local
MCP 端口：8931
远程地址：<REMOTE_USER>@<REMOTE_HOST>
ChatGPT 对话：<CHATGPT_CONVERSATION_URL>
```

文中的尖括号内容都是占位符，必须替换。PowerShell 命令不要原样粘贴未替换的 `<REMOTE_USER>` 等字样；下文用带引号的变量示范目标填写。

如果 `8931` 已被占用，可以换成其他高位端口，但以下位置必须同步修改：

1. Playwright MCP 的 `--port`；
2. `--allowed-hosts` 中的端口；
3. SSH `-R` 两侧端口；
4. Codex MCP URL。

## 4. 首次配置或完整恢复

恢复时需要同时保留两个 Windows PowerShell 窗口：一个运行 MCP，一个运行 SSH 隧道。

### 4.1 关闭 VS Code 对 8931 的普通端口转发

在当前 Remote-SSH 窗口的命令面板中打开 `Preferences: Open Remote Settings (JSON)`，把以下配置合并到现有设置；如果已有 `remote.portsAttributes`，只增加或更新其中的 `8931`，保留其他端口配置：

```json
{
  "remote.portsAttributes": {
    "8931": {
      "onAutoForward": "ignore"
    }
  }
}
```

这只排除 8931 的自动转发。然后在 VS Code 的 `PORTS / 端口` 面板找到现有的 `8931`，右键选择 `Stop Forwarding / 停止转发`。**不能只改设置而保留已经存在的转发。** Windows 的 8931 必须留给 Playwright MCP。

如果重新连接后旧转发又恢复，检查 `Remote: Restore Forwarded Ports`；不需要自动恢复其他转发时，可在同一设置作用域设 `"remote.restoreForwardedPorts": false`。若不需要任何自动发现转发，也可设 `"remote.autoForwardPorts": false`，但官方说明它不能禁止扩展或打开 URL 触发的所有转发。优先保留逐端口排除，并检查现有 PORTS 条目。

远程设置可能被 Workspace 设置覆盖，设置未生效时核对该层。上述选项分别依据 [VS Code 官方配置定义](https://github.com/microsoft/vscode/blob/main/src/vs/workbench/contrib/remote/common/remote.contribution.ts) 和 [Remote-SSH 文档](https://code.visualstudio.com/docs/remote/ssh)。

原因：VS Code 的常规端口转发方向与这里需要的 SSH 反向隧道不同。曾经出现过 Windows 的 `8931` 被 `Code.exe` 占用，随后 SSH 又把该端口反向转回服务器，形成无响应的转发环路。

可在 Windows PowerShell 检查端口所有者：

```powershell
Get-NetTCPConnection -LocalPort 8931 -State Listen | ForEach-Object { Get-Process -Id $_.OwningProcess | Select-Object Id,ProcessName,Path }
```

正常情况下，启动 MCP 后端口应由 Node.js 进程持有，而不是 VS Code。

### 4.2 在 Windows 启动 Playwright MCP

在第一个 Windows PowerShell 中运行：

```powershell
npx.cmd -y @playwright/mcp@latest --browser msedge --host 127.0.0.1 --port 8931 --allowed-hosts "localhost:8931,127.0.0.1:8931" --user-data-dir "$env:LOCALAPPDATA\CodexEdgeProfile"
```

保持这个窗口运行。

说明：

- `--host 127.0.0.1` 只监听 Windows 回环地址，不暴露到局域网。
- `--allowed-hosts` 必须匹配完整 HTTP `Host`，包括 `:8931`。缺少端口会返回 `403 Forbidden`。
- `--user-data-dir` 使用独立 profile，避免操作日常 Edge 配置，也能保留 ChatGPT 登录状态。
- 首次执行 `npx.cmd ...@latest` 会下载 Playwright MCP 包。
- 一个持久化 profile 同时只能被一个浏览器实例使用。出现 profile lock 时，先关闭旧 MCP/Edge 进程。

如果希望固定版本以提高可复现性，可把 `@latest` 换成经过验证的具体版本；升级后应重新执行本手册的连通性检查。

### 4.3 在 Windows 验证 MCP 服务

另开一个 PowerShell：

```powershell
curl.exe --noproxy "*" --max-time 5 -i http://127.0.0.1:8931/mcp
```

结果解释：

- `400 Bad Request` / `Invalid request`：通常是正常的，说明 HTTP 服务与 Host 校验都已通过，只是普通 GET 没有执行 MCP 初始化。
- `403 Forbidden`：检查 `--allowed-hosts` 是否包含 `127.0.0.1:8931`，端口不能省略。
- `Connection refused`：MCP 没启动、已退出或端口写错。
- 连接成功但一直超时：检查端口是否被 VS Code 或其他转发进程占用。

不要为了消除 403 使用 `--allowed-hosts "*"`。本方案只需要允许两个明确的本机 Host。

### 4.4 建立 Windows → 远程服务器的 SSH 反向隧道

先找到**平时已经能 SSH 登录的同一个目标**：它可能是 VS Code Remote-SSH 的 Host 别名，也可能是完整的 `user@host`。在 Windows PowerShell 中运行下面的输入命令，然后输入真实目标；不要输入示例占位符：

```powershell
$SshTarget = Read-Host "输入平时可登录的 SSH 目标（Host 别名或 user@host）"
ssh -G $SshTarget | Select-String '^(user|hostname|port|remoteforward|localforward|proxyjump|proxycommand|identityfile) '
```

如果平时必须指定 `-F`、`-p` 或 `-i`，本节检查和隧道命令也必须保留相同参数。先做一次普通 SSH 检查，暂时清除转发请求，以区分登录失败与端口转发失败：

```powershell
ssh -o ClearAllForwardings=yes $SshTarget hostname
```

此命令应返回服务器主机名并退出。失败时先按第 8.10 节定位登录问题。`ClearAllForwardings=yes` 只用于这个登录诊断，它也会清除命令行上的 `-R`，**不能带到下面的隧道命令**。

核对 `ssh -G` 输出中没有不需要的旧转发后，在完成登录检查的**同一个 PowerShell 窗口**运行下面这一整行，并保持窗口打开。沿用已经验证的 `$SshTarget`，不要重新赋成示例占位符；另开窗口时变量不会自动继承，须重新输入真实目标。

只复制代码内容，不要复制终端的 `PS C:\...>` 或 `>>` 提示符。`>>` 是 PowerShell 自动显示的续行提示，粘贴进命令会被解释为重定向。若当前已处于错误的续行输入中，先按 `Ctrl+C`，再粘贴：

```powershell
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R 127.0.0.1:8931:127.0.0.1:8931 $SshTarget
```

光标停住、没有 shell 提示符通常就是成功状态，因为：

- `-N` 表示不执行远程命令；
- `-T` 表示不分配终端；
- `-R` 请求把服务器回环端口反向转到 Windows MCP。

这仍不证明浏览器服务可达，必须执行第 4.5 节的 HTTP 检查。`ExitOnForwardFailure` 只检查转发建立，不能保证最终目标服务正常。

#### 已确认继承了无关转发时

以前出现过命令要求转发 `8931`，错误却报告远端 `31080` 的情况，原因是 SSH config 额外加载了旧 `RemoteForward`。这只是已知历史案例，不能直接当作此次故障原因。

可手工创建独立文件 `%USERPROFILE%\.ssh\codex-edge-tunnel.conf`，只复制已验证的登录字段，不复制旧 `RemoteForward` / `LocalForward`。内容示例（全部占位符须替换；未使用的可选项不填写）：

```sshconfig
Host codex-edge-tunnel
    HostName REAL_REMOTE_HOST
    User REAL_REMOTE_USER
    Port REAL_SSH_PORT
    # 仅在原连接需要密钥时填写：
    # IdentityFile C:/Users/LOCAL_USER/.ssh/KEY_FILE
    # 仅在原连接需要跳板时填写，并在本文件定义对应 Host：
    # ProxyJump jump-host
```

然后使用：

```powershell
ssh -F "$env:USERPROFILE\.ssh\codex-edge-tunnel.conf" -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R 127.0.0.1:8931:127.0.0.1:8931 codex-edge-tunnel
```

如果原连接包含 ProxyCommand、证书或特殊认证，也要保留所需配置，不能只照抄上面的简化示例。

`-F none` 会忽略所有 SSH 配置文件，包括 Host 别名、非默认端口、IdentityFile 和跳板设置。因此只有已经显式掌握全部必需参数时才使用，例如无跳板、密钥登录的情况：

```powershell
$SshDestination = "REAL_USER@REAL_HOST"
$SshPort = 22 # 替换为真实 SSH 端口
$SshIdentity = "C:\Users\LOCAL_USER\.ssh\KEY_FILE" # 替换为实际密钥文件路径
ssh -F none -p $SshPort -i $SshIdentity -N -T -o ExitOnForwardFailure=yes -R 127.0.0.1:8931:127.0.0.1:8931 $SshDestination
```

不要把私钥内容、密码或令牌写入文档或仓库。

### 4.5 在远程服务器验证隧道

在远程 Linux 终端运行：

```bash
curl --noproxy '*' --connect-timeout 3 --max-time 5 \
  -i http://127.0.0.1:8931/mcp
```

和 Windows 本地检查一样，普通 GET 返回 `400` 通常表示链路已经贯通。以下结果说明尚未成功：

- `Connection refused`：SSH 隧道没有运行，或 `-R` 建立失败。
- 超时且无 HTTP 响应：检查 Windows MCP、VS Code 端口环路和代理变量。
- `403`：Windows MCP 的 `--allowed-hosts` 不正确。

在服务器检查监听：

```bash
ss -ltnp '( sport = :8931 )'
```

监听地址应为回环地址；若出现 `0.0.0.0:8931` 或 `[::]:8931`，先停止本次隧道并让管理员核查 `GatewayPorts`。服务端 `GatewayPorts yes` 可以强制通配地址监听，即使命令请求了回环地址。不要看到旧监听就直接杀进程；先确认它是否属于旧 SSH 隧道或其他用户。

### 4.6 配置 Codex MCP（不要求终端安装 Codex）

#### 推荐：通过 IDE 编辑正在生效的 config.toml

在当前 Codex IDE 扩展中打开齿轮菜单，选择 `Codex Settings → Open config.toml`（不同版本文案可能略有区别）。官方默认用户配置是 `~/.codex/config.toml`；若设置了自定义 Codex 配置目录，以当前实例实际打开的文件为准。

本文假设 Codex 后端运行在服务器上，所以应该编辑该后端读取的配置。检查编辑器显示的文件所属主机；Remote-SSH 工作区不应仅凭窗口外观判断配置归属。若实际 Codex 后端在 Windows，`127.0.0.1` 指 Windows，此时网络拓扑与本文不同。

在现有文件中增加或修改这个表，其他配置保留：

```toml
[mcp_servers.edge-local]
url = "http://127.0.0.1:8931/mcp"
enabled = true
```

如果已有 `[mcp_servers.edge-local]`，修改该表，不要再追加同名表。不要用这三行覆盖整个配置文件。独立 HTTP MCP 不需要填写 `command = "codex"`。

官方说明 CLI 和 IDE 扩展支持同一配置格式，也支持直接编辑 TOML，因此这里不需要执行 `codex mcp add`。[MCP 文档](https://developers.openai.com/codex/mcp)、[配置入口](https://developers.openai.com/codex/config-basic/)

保存后按第 4.7 节重载会话，并从扩展 MCP 设置/工具列表检查 `edge-local`。**配置条目存在只证明配置已写入；工具实际可用才证明当前客户端连接成功。**

#### 可选：终端确实有 codex 时

先确认命令可用，再使用 CLI 管理同一个配置：

```bash
command -v codex
codex mcp list
codex mcp get edge-local
```

如果不存在，再执行一次：

```bash
codex mcp add edge-local --url http://127.0.0.1:8931/mcp
codex mcp list
```

预期列表中出现：

```text
edge-local  http://127.0.0.1:8931/mcp  enabled
```

没有 `codex` 命令时跳过这一小节，使用上面的 TOML 方式。后续恢复连接一般无需重复添加配置。

如果 URL 或端口发生变化，先明确移除旧条目再添加：

```bash
codex mcp remove edge-local
codex mcp add edge-local --url 'http://127.0.0.1:<NEW_PORT>/mcp'
```

### 4.7 重启 Codex 或 IDE 扩展

Codex 通常在会话启动时建立工具目录。刚注册 MCP 后：

1. 在 VS Code 命令面板执行 `Developer: Reload Window`，或重启 Codex 扩展；
2. 新建一个 Codex 会话；
3. 在 IDE 的 MCP 设置和会话工具列表中检查服务；仅 CLI/TUI 用户使用 `/mcp`；
4. 确认 `edge-local` 的浏览器工具可见。

仅修改配置但不重启旧会话，可能导致当前会话仍看不到新工具。

### 4.8 首次打开 Edge 并登录 ChatGPT

给 Codex 一个限制明确的指令，例如：

```text
使用 edge-local 打开 https://chatgpt.com。
等待我在 Windows 的专用 Edge 窗口中手工登录。
登录完成前不要读取其他页面，也不要发送任何消息。
```

Playwright 第一次执行浏览器工具时会打开 Edge。随后在 Windows 中手工完成：

- ChatGPT 登录；
- 验证码；
- 双因素认证；
- 必要的隐私或 Cookie 提示。

登录后再给出目标会话 URL：

```text
打开 <CHATGPT_CONVERSATION_URL>，只读取并概括该对话，不发送消息。
```

如果确实要让 Codex 向网页端发送信息，必须单独明确授权，并限定：

- 目标会话；
- 要发送的完整文字；
- 是否允许直接点击发送；
- 是否需要发送前再次展示草稿。

## 5. 日常恢复：最短操作顺序

只要远程 Codex 配置和 Windows profile 仍在，后续通常只需：

1. 确认 VS Code 没有转发 `8931`。
2. Windows PowerShell A：启动 Playwright MCP。
3. Windows PowerShell B：使用第 4.4 节已经验证的 SSH 登录配置启动 `-R` 隧道。
4. 服务器：用 `curl` 确认 `127.0.0.1:8931/mcp` 返回 HTTP 响应。
5. Codex IDE：确认实际生效配置有 `edge-local`，重载后在 MCP 设置/工具列表确认可用；有 CLI 时可辅助执行 `codex mcp list`。
6. 重启 Codex 会话，让工具目录重新加载。
7. 让 Codex 打开目标 ChatGPT 页面；必要时在 Edge 手工重新登录。

不需要重复注册 MCP，也不应保存上次的 `Mcp-Session-Id`。

## 6. 推荐的浏览器操作方式

优先使用 Playwright MCP 暴露的高层工具：

1. `browser_navigate` 打开页面；
2. `browser_snapshot` 读取可访问性树；
3. 根据最新 snapshot 中的元素引用进行 `browser_click`、`browser_type`；
4. 等待网页端生成完成后再读取结果。

工具名称可能随 Playwright MCP 版本变化，应以当前 MCP 的 `tools/list` 或 Codex 工具目录为准。

避免长期依赖这些脆弱选择器：

```text
button[aria-label="发送"]
[contenteditable="true"][role="textbox"]
```

它们在之前的中文 ChatGPT 页面上工作过，但页面语言、DOM 或 aria-label 更新后可能失效。优先依据最新 accessibility snapshot 操作。

### 读取长对话时的网页提取思路

如果普通 snapshot 被截断，可使用 `browser_evaluate` 在当前页面读取 `main` 的可见文字。下面只是应急思路，不保证未来 DOM 兼容：

```javascript
() => {
  const main = document.querySelector("main");
  if (!main) return { busy: false, text: "", error: "main not found" };

  const copy = main.cloneNode(true);
  copy.querySelectorAll(".katex").forEach((node) => {
    const annotation = node.querySelector("annotation");
    if (annotation) {
      node.replaceWith(document.createTextNode(` $${annotation.textContent}$ `));
    }
  });

  return {
    busy: Boolean(document.querySelector('button[aria-label="停止"]')),
    text: copy.innerText || copy.textContent,
  };
}
```

注意：这会读取当前 `main` 中可能存在的整段对话。只在用户明确指定的会话中使用，并控制输出范围。

## 7. 当前 Codex 看不到工具时的原始 RPC 应急方法

首选方案始终是重启 Codex/IDE，让 `edge-local` 正常出现在工具目录中。只有旧会话无法重启、但确实需要验证 MCP 时，才直接调用 Streamable HTTP MCP。

### 7.1 重要规则

- 每次 MCP 服务或隧道重启后重新执行 `initialize`。
- 从初始化响应头读取新的 `Mcp-Session-Id`。
- Session ID 是临时值，不写入脚本、文档、Git 或聊天记录。
- 先调用 `tools/list`，不要假定工具名称与旧版本相同。

### 7.2 初始化请求示例

在远程服务器执行：

```bash
MCP_HEADERS_FILE="$(mktemp)"
MCP_BODY_FILE="$(mktemp)"

curl --noproxy '*' --connect-timeout 3 --max-time 20 \
  -D "$MCP_HEADERS_FILE" \
  -o "$MCP_BODY_FILE" \
  -X POST http://127.0.0.1:8931/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  --data-raw '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"manual-connectivity-check","version":"1.0"}}}'

MCP_SESSION_ID="$(tr -d '\r' < "$MCP_HEADERS_FILE" | awk 'tolower($1) == "mcp-session-id:" {print $2}' | tail -n 1)"

test -n "$MCP_SESSION_ID" || {
  echo 'No Mcp-Session-Id returned'
  sed -n '1,80p' "$MCP_HEADERS_FILE"
  exit 1
}

sed -n '1,120p' "$MCP_BODY_FILE"
```

随后通知服务器初始化完成：

```bash
curl --noproxy '*' --max-time 10 \
  -X POST http://127.0.0.1:8931/mcp \
  -H "Mcp-Session-Id: $MCP_SESSION_ID" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  --data-raw '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}'
```

列出当前工具：

```bash
curl --noproxy '*' --max-time 20 \
  -X POST http://127.0.0.1:8931/mcp \
  -H "Mcp-Session-Id: $MCP_SESSION_ID" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  --data-raw '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
```

根据 `tools/list` 返回的名称调用工具。例如，只有列表中确实存在 `browser_navigate` 时才执行：

```bash
curl --noproxy '*' --max-time 60 \
  -X POST http://127.0.0.1:8931/mcp \
  -H "Mcp-Session-Id: $MCP_SESSION_ID" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  --data-raw '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"browser_navigate","arguments":{"url":"https://chatgpt.com"}}}'
```

使用完成后删除临时响应文件并清空当前 shell 变量：

```bash
rm -f -- "$MCP_HEADERS_FILE" "$MCP_BODY_FILE"
unset MCP_HEADERS_FILE MCP_BODY_FILE MCP_SESSION_ID
```

这里删除的只是本次初始化产生的临时 HTTP 文件，不会删除浏览器 profile 或 Codex 配置。

## 8. 常见故障与根因

### 8.1 `remote port forwarding failed for listen port 31080`

症状：命令只写了 `8931`，报错却提到另一个端口。

历史案例的根因：SSH config 中继承了旧的 `RemoteForward`，`ExitOnForwardFailure=yes` 因任一转发失败而退出。新故障仍须核实实际生效配置。

处理：

```powershell
ssh -G $SshTarget |
    Select-String '^(user|hostname|port|remoteforward|proxyjump|proxycommand|identityfile) '
```

确认后使用第 4.4 节的独立配置文件，或在完整掌握连接参数后使用 `-F none`；不要丢失原本需要的登录配置。

### 8.2 SSH 窗口只有闪烁光标

使用 `-N -T` 时这是正常表现。只要没有退出或报错，并且服务器 `curl` 能得到响应，就表示隧道正在运行。

### 8.3 Windows 与服务器 curl 都超时

先看 Windows `8931` 的监听进程。如果是 `Code.exe`，按第 4.1 节排除 8931 自动转发并停止已有转发，再启动或重启原 MCP 进程（不要同时启动多个实例）。仍在运行的 SSH 反向隧道可以保留。先在 Windows 用 `curl.exe` 验证，再在服务器用 `curl` 验证。若监听者是 Node.js，查看 MCP 窗口是否有错误或 profile lock。

### 8.4 Windows 返回 403

`--allowed-hosts` 匹配的是带端口的完整 Host。使用：

```text
localhost:8931,127.0.0.1:8931
```

而不是只写 `localhost,127.0.0.1`。

### 8.5 Windows 返回 400 / Invalid request

普通 GET 下通常正常，说明服务可达。用 Codex MCP 客户端或第 7 节的标准 `initialize` POST 做协议验证。

### 8.6 服务器 Connection refused

检查：

1. SSH 隧道 PowerShell 是否仍在运行；
2. `-R` 是否建立成功；
3. 远端 `8931` 是否被旧进程占用；
4. 是否因为网络中断导致 SSH 退出。

### 8.7 `edge-local` 已注册，但当前 Codex 没有浏览器工具

Codex 会话可能在配置前已经启动。先确认编辑的是当前后端读取的配置，再重载 IDE/扩展并新建会话。配置表存在或 `codex mcp list` 有条目都不证明当前旧会话已经加载工具。

### 8.8 ChatGPT 页面只显示登录页

当前 Edge profile 未登录、登录过期，或者打开了错误的 profile。回到 Windows 的专用 Edge 手工登录。不要尝试从服务器复制 Cookie。

### 8.9 ChatGPT 回复尚未完成

先检查页面是否仍显示“停止”按钮或其他生成中状态。不要在输出流未完成时重复读取或发送下一条消息。

### 8.10 本次隧道失败：按报错所在层定位

2026-10-18 用户日志已确认：普通 SSH 的 `hostname` 返回成功；把目标重新设为 `YOUR_EXISTING_SSH_ALIAS` 后发生主机名解析失败；将终端的 `>>` 提示符复制进命令后发生 `StreamAlreadyRedirected` 解析错误，SSH 尚未执行。配置还包含另一个 `31080 → localhost:7897` 反向转发，但本次日志未证明它发生了端口冲突。下表其他情况仍是待排查分支。隧道命令必须在 **Windows 本机 PowerShell** 运行。

| 原始错误或表现 | 可检查的原因 | 下一项证据 |
| --- | --- | --- |
| PowerShell parser error、`StreamAlreadyRedirected`、`-R` 被当成命令 | 复制了 `PS ...>` / `>>` 提示符、命令断行/倒序或占位符未替换 | `Ctrl+C` 退出错误输入，使用第 4.4 节单行形式，只复制代码内容 |
| `Could not resolve hostname` | 别名拼错；`-F none` 忽略了 HostName；DNS 问题 | `ssh -G` 中的 hostname，与原本可用连接对照 |
| 连接 SSH 端口超时/拒绝 | 地址/端口不对、VPN/跳板/网络/sshd 问题 | 先恢复原有 SSH 登录；不是修改 MCP 参数 |
| `Permission denied (publickey...)` | 用户、密钥、证书或认证配置不一致 | 原有可用登录命令与 `ssh -G` 的 user/identityfile |
| `remote port forwarding failed ...8931` | 远端端口占用或服务端禁止转发 | 远端 `ss`、客户端 `ssh -vv` 日志、管理员转发策略 |
| 报错端口不是 8931 | 可能继承了其他 Forward | `ssh -G` 的 remoteforward/localforward |
| `administratively prohibited` | 转发策略或密钥级限制 | 管理员核查有效的 `AllowTcpForwarding`、`DisableForwarding`、`PermitListen`、Match 块及 authorized_keys 限制 |
| SSH 不退出，但 HTTP 超时/拒绝 | 隧道之后的服务没启动、指向错误或转发环路 | 先测 Windows MCP，再测远端 HTTP |

需要更详细日志时，在已经核对的同一隧道命令中增加 `-vv`，只复现一次。分享命令和报错时替换用户名、主机地址与私人路径；不要提供密钥内容、密码、Cookie 或整个 Codex 配置文件。禁止用关闭主机身份检查、开放公网 MCP 或擅改服务器 SSH 安全策略来绕过故障。

### 8.11 `codex: command not found`

采用第 4.6 节的 IDE/TOML 配置路径。SSH 隧道由 Windows 的 `ssh` 建立，本身与远程服务器是否安装 Codex CLI 无关；两个问题需要分层诊断。

## 9. 安全边界

- MCP 只绑定 Windows `127.0.0.1`；SSH `-R` 请求服务器 `127.0.0.1`，并用 `ss` 验证实际绑定，避免服务端 `GatewayPorts` 覆盖。
- 不把服务绑定到 `0.0.0.0`，不在路由器或云安全组开放 `8931`。
- 不使用 `--allowed-hosts '*'`。
- `%LOCALAPPDATA%\CodexEdgeProfile` 包含登录状态，不上传、不备份到公开位置、不提交 Git。
- 不保存或复用 `Mcp-Session-Id`；它不是持久配置。
- 在多人服务器上，其他本机用户可能访问回环端口。这套无认证隧道只适合可信服务器；否则应增加经过审查的认证层。
- 默认只授予 Codex 只读任务。发送消息、删除内容、修改账号设置等写操作需要用户逐项明确授权。
- 网页内容本身可能包含提示注入。不要把网页中的指令自动提升为系统命令或仓库操作。
- 不让浏览器访问与任务无关的私人标签页、邮件、云盘或账号设置。

## 10. 停止与清理

临时停止访问：

1. 在 SSH 隧道 PowerShell 中按 `Ctrl+C`；
2. 在 Playwright MCP PowerShell 中按 `Ctrl+C`；
3. 关闭专用 Edge 窗口。

这样会断开远程访问，但保留：

- 服务器上的 `edge-local` Codex 配置；
- Windows 独立 Edge profile 中的登录状态。

如果以后不再使用，在 IDE 打开实际配置文件，把该表设为 `enabled = false`，或仅删除 `[mcp_servers.edge-local]` 及其字段。不要删除其他 MCP 或账号配置。有 CLI 时也可使用：

```bash
codex mcp remove edge-local
```

然后在 Windows 确认不再需要登录状态后，手工删除：

```text
%LOCALAPPDATA%\CodexEdgeProfile
```

删除 profile 会清除该专用浏览器中的登录状态，无法自动恢复；执行前应再次确认。

## 11. 一页恢复清单

```text
[ ] VS Code PORTS 中没有 8931 普通转发
[ ] Windows MCP PowerShell 正在运行
[ ] Windows curl 127.0.0.1:8931/mcp 得到 HTTP 响应
[ ] Windows SSH -R PowerShell 正在运行
[ ] 服务器 curl 127.0.0.1:8931/mcp 得到 HTTP 响应
[ ] Codex 实际配置有 edge-local；不要求终端有 codex 命令
[ ] Codex/IDE 已在 MCP 可用后重新启动
[ ] 当前会话能看到 edge-local 浏览器工具
[ ] 专用 Edge profile 已登录 ChatGPT
[ ] Codex 只访问用户明确指定的会话
[ ] 发送消息等写操作已获得单独授权
```

连通性与登录检查通过后，远程 Codex 才能尝试读取指定的 ChatGPT 网页会话；最后两项是必须持续遵守的权限边界。
