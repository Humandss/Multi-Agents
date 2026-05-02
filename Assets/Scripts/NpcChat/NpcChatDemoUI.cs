using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace NpcChat
{
    /// <summary>
    /// 데모 UI — NPC와 대화 + 시간 진행으로 정보 전파.
    ///
    /// Hierarchy 예시:
    ///   Canvas
    ///   ├── NpcDropdown   (TMP_Dropdown: elias/hermann/mathilda/finn/bernhardt)
    ///   ├── ChatLog       (ScrollView 안 TMP_Text)
    ///   ├── InputField    (TMP_InputField)
    ///   ├── SendButton    (Button)
    ///   ├── TimeButton    (Button "시간 진행")  ← NEW
    ///   ├── DayText       (TMP_Text "Day 0")    ← NEW (선택)
    ///   └── StatusText    (TMP_Text)
    /// </summary>
    public class NpcChatDemoUI : MonoBehaviour
    {
        [Header("Server")]
        public string serverHost = "127.0.0.1";
        public int serverPort = 8000;

        [Header("UI — Required")]
        public TMP_Dropdown npcDropdown;
        public TMP_Text chatLog;
        public TMP_InputField inputField;
        public Button sendButton;
        public TMP_Text statusText;

        [Header("UI — Time Advance (Optional)")]
        public Button timeButton;
        public TMP_Text dayText;

        [Header("Settings")]
        public bool showRetrievedMemories = true;
        public bool showTickEvents = true;

        private NpcChatClient _client;
        private readonly StringBuilder _log = new StringBuilder();
        private int _currentDay;

        private void Start()
        {
            sendButton.onClick.AddListener(OnSendClicked);
            inputField.onSubmit.AddListener(_ => OnSendClicked());
            npcDropdown.onValueChanged.AddListener(OnNpcChanged);
            if (timeButton != null) timeButton.onClick.AddListener(OnTimeClicked);

            UpdateDayText();
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
            _client.OnTickEvents += HandleTickEvents;
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

        private void OnTimeClicked()
        {
            if (_client == null || !_client.IsOpen)
            {
                SetStatus("연결되지 않음");
                return;
            }
            SetStatus("시간 진행 중... (전파 시뮬레이션)");
            _ = _client.SendTimeAdvanceAsync();
        }

        private void HandleResponse(ServerMessage resp)
        {
            string npcName = string.IsNullOrEmpty(resp.npc) ? "NPC" : resp.npc;
            AppendLog($"<color=#fc9>{npcName}:</color> {resp.text}");

            if (showRetrievedMemories && resp.memories_used != null && resp.memories_used.Length > 0)
            {
                AppendLog($"<size=80%><color=#888>  ↳ 회상 {resp.memories_used.Length}개:</color></size>");
                foreach (var m in resp.memories_used)
                {
                    AppendLog($"<size=80%><color=#888>     [{m.importance}/{m.source}] {Truncate(m.text, 60)}</color></size>");
                }
            }

            SetStatus($"{npcName} 응답 완료 ({resp.latency_ms}ms)");
        }

        private void HandleTickEvents(ServerMessage tick)
        {
            _currentDay = tick.day;
            UpdateDayText();

            int n = tick.events != null ? tick.events.Length : 0;
            AppendLog($"<size=85%><color=#fb9>━━━ Day {tick.day} 진행 ({n}개 정보 전달) ━━━</color></size>");

            if (showTickEvents && tick.events != null)
            {
                int max = Mathf.Min(tick.events.Length, 8);
                for (int i = 0; i < max; i++)
                {
                    var ev = tick.events[i];
                    AppendLog(
                        $"<size=80%><color=#aaa>  • {ev.from} → {ev.to}: " +
                        $"\"{Truncate(ev.transformed, 40)}\"</color></size>"
                    );
                }
                if (tick.events.Length > max)
                {
                    AppendLog($"<size=80%><color=#aaa>  • (외 {tick.events.Length - max}건)</color></size>");
                }
            }
            SetStatus($"Day {tick.day} 완료 ({n}개 전달)");
        }

        private void UpdateDayText()
        {
            if (dayText != null) dayText.text = $"Day {_currentDay}";
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
