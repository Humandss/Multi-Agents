using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace NpcChat
{
    /// <summary>
    /// 간단 데모 UI — 1개 NPC와 대화.
    ///
    /// Hierarchy 예시:
    ///   Canvas
    ///   ├── NpcDropdown   (TMP_Dropdown, 옵션: elias / hermann / mathilda / finn / bernhardt)
    ///   ├── ChatLog       (ScrollView 안에 TMP_Text)
    ///   ├── InputField    (TMP_InputField)
    ///   ├── SendButton    (Button)
    ///   └── StatusText    (TMP_Text)
    ///
    /// 이 스크립트를 빈 GameObject에 붙이고 위 컴포넌트들을 인스펙터에 연결.
    /// </summary>
    public class NpcChatDemoUI : MonoBehaviour
    {
        [Header("Server")]
        public string serverHost = "127.0.0.1";
        public int serverPort = 8000;

        [Header("UI")]
        public TMP_Dropdown npcDropdown;
        public TMP_Text chatLog;
        public TMP_InputField inputField;
        public Button sendButton;
        public TMP_Text statusText;

        [Header("Settings")]
        public bool showRetrievedMemories = true;

        private NpcChatClient _client;
        private readonly StringBuilder _log = new StringBuilder();

        private void Start()
        {
            sendButton.onClick.AddListener(OnSendClicked);
            inputField.onSubmit.AddListener(_ => OnSendClicked());
            npcDropdown.onValueChanged.AddListener(OnNpcChanged);

            // 초기 NPC 연결
            _ = ConnectToCurrent();
        }

        private void Update()
        {
            _client?.DispatchQueue();
        }

        private async System.Threading.Tasks.Task ConnectToCurrent()
        {
            if (_client != null) await _client.CloseAsync();

            string npc = npcDropdown.options[npcDropdown.value].text;
            SetStatus($"{npc} 연결 중...");

            _client = new NpcChatClient(npc)
            {
                ServerHost = serverHost,
                ServerPort = serverPort,
            };
            _client.OnReady += () => SetStatus($"{npc} 연결됨");
            _client.OnResponse += HandleResponse;
            _client.OnError += msg => SetStatus($"에러: {msg}");
            _client.OnClosed += () => SetStatus("연결 종료");

            await _client.ConnectAsync();
        }

        private void OnNpcChanged(int idx)
        {
            ClearLog();
            _ = ConnectToCurrent();
        }

        private void OnSendClicked()
        {
            string text = inputField.text.Trim();
            if (string.IsNullOrEmpty(text)) return;
            inputField.text = "";

            AppendLog($"<color=#9bd>플레이어:</color> {text}");
            SetStatus("응답 생성 중...");
            _ = _client.SendChatAsync(text);
        }

        private void HandleResponse(ChatResponse resp)
        {
            string npcName = string.IsNullOrEmpty(resp.npc) ? "NPC" : resp.npc;
            AppendLog($"<color=#fc9>{npcName}:</color> {resp.text}");

            if (showRetrievedMemories && resp.memories_used != null && resp.memories_used.Length > 0)
            {
                AppendLog($"<size=80%><color=#888>  ↳ 회상 {resp.memories_used.Length}개:");
                foreach (var m in resp.memories_used)
                {
                    AppendLog($"<size=80%><color=#888>     [{m.importance}/{m.source}] {Truncate(m.text, 60)}</color></size>");
                }
            }

            SetStatus($"{npcName} 응답 완료 ({resp.latency_ms}ms)");
        }

        private void AppendLog(string line)
        {
            _log.AppendLine(line);
            if (chatLog != null) chatLog.text = _log.ToString();
        }

        private void ClearLog()
        {
            _log.Clear();
            if (chatLog != null) chatLog.text = "";
        }

        private void SetStatus(string s)
        {
            if (statusText != null) statusText.text = s;
        }

        private string Truncate(string s, int n) => s.Length <= n ? s : s.Substring(0, n) + "…";

        private async void OnApplicationQuit()
        {
            if (_client != null) await _client.CloseAsync();
        }
    }
}
