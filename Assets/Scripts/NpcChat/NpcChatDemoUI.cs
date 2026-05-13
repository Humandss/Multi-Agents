using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace NpcChat
{
    /// <summary>
    /// 게임 NPC 대화창 UI 매니저.
    ///
    /// 기능:
    /// - 현재 응답 (NPC 이름 + 본문) 큰 영역
    /// - 대화 history 누적 로그 (스크롤)
    /// - 회상 메모리 hint
    /// - Quest 카드
    /// - 연결 상태 + 응답 대기 표시
    /// - Day 진행 + 정보 전파 이벤트 표시
    /// - NPC별 색상
    /// - 타이핑 효과 (선택)
    ///
    /// 동작 모드:
    /// - autoConnectViaDialogueManager=true (default): F키로 NPC 가까이서 trigger
    /// - false: Dropdown으로 직접 선택 (개발자 데모용)
    /// </summary>
    public class NpcChatDemoUI : MonoBehaviour
    {
        // ========== Inspector ==========
        [Header("Server")]
        public string serverHost = "127.0.0.1";
        public int serverPort = 8000;

        [Header("Current Response (대화창)")]
        [Tooltip("NPC 이름 — 큰 글씨")]
        public TMP_Text currentNpcNameText;
        [Tooltip("NPC 응답 본문 — 메인 영역")]
        public TMP_Text currentResponseText;

        [Header("Input")]
        public TMP_InputField inputField;
        public Button sendButton;

        [Header("History Log (선택)")]
        [Tooltip("이전 대화 누적 — 비워두면 history 표시 안 함")]
        public TMP_Text historyLogText;
        [Tooltip("history 스크롤 (선택)")]
        public ScrollRect historyScrollRect;

        [Header("Status / Memory")]
        public TMP_Text statusText;
        [Tooltip("회상 메모리 hint (작은 텍스트)")]
        public TMP_Text memoryHintText;

        [Header("Quest Card (선택)")]
        public GameObject questCard;
        public TMP_Text questTitleText;
        public TMP_Text questDescText;
        public TMP_Text questRewardText;

        [Header("Time Advance (Day 진행, 선택)")]
        public Button timeButton;
        public TMP_Text dayText;

        [Header("Dropdown (개발자 모드용, 선택)")]
        [Tooltip("Auto mode에서는 자동 숨김")]
        public TMP_Dropdown npcDropdown;

        [Header("Game Flow")]
        [Tooltip("true: DialogueManager 이벤트로 NPC 자동 결정 (F키 트리거)")]
        public bool autoConnectViaDialogueManager = true;
        [Tooltip("Auto 모드 시 Dropdown 숨김")]
        public bool hideDropdownInAutoMode = true;

        [Header("Settings")]
        public bool showRetrievedMemories = true;
        public bool showTickEvents = true;
        public bool keepHistoryLog = true;
        [Tooltip("타이핑 효과 (응답 한 글자씩 표시)")]
        public bool useTypewriterEffect = true;
        [Tooltip("글자당 지연 (초)")]
        public float typewriterCharDelay = 0.02f;

        [Header("NPC Colors")]
        public NpcColorEntry[] npcColors = new[]
        {
            new NpcColorEntry { npc = "elias",     color = new Color(0.6f, 0.75f, 1f) },
            new NpcColorEntry { npc = "hermann",   color = new Color(0.85f, 0.85f, 0.85f) },
            new NpcColorEntry { npc = "mathilda",  color = new Color(1f, 0.7f, 0.5f) },
            new NpcColorEntry { npc = "finn",      color = new Color(1f, 0.95f, 0.6f) },
            new NpcColorEntry { npc = "bernhardt", color = new Color(0.85f, 0.7f, 1f) },
        };
        public Color playerColor = new Color(0.6f, 0.85f, 1f);
        public Color systemColor = new Color(0.7f, 0.7f, 0.7f);

        [System.Serializable]
        public class NpcColorEntry { public string npc; public Color color; }

        // ========== 내부 ==========
        private NpcChatClient _client;
        private readonly StringBuilder _log = new StringBuilder();
        private int _currentDay;
        private bool _subscribed = false;
        private bool _waitingResponse = false;
        private Coroutine _typewriterRoutine;
        private Dictionary<string, Color> _colorMap;

        // ========== Unity 라이프사이클 ==========
        private void Awake()
        {
            BuildColorMap();
            if (sendButton != null) sendButton.onClick.AddListener(OnSendClicked);
            if (inputField != null) inputField.onSubmit.AddListener(_ => OnSendClicked());
            if (npcDropdown != null) npcDropdown.onValueChanged.AddListener(OnNpcChanged);
            if (timeButton != null) timeButton.onClick.AddListener(OnTimeClicked);
            if (questCard != null) questCard.SetActive(false);

            AutoWireMissingSlots();
        }

        /// <summary>
        /// Inspector에서 슬롯이 미연결된 경우 Scene 탐색으로 자동 연결.
        /// historyLogText / historyScrollRect 가 흔히 누락됨.
        /// </summary>
        private void AutoWireMissingSlots()
        {
            // 1) 자식에서 ScrollRect 검색
            if (historyScrollRect == null)
            {
                historyScrollRect = GetComponentInChildren<ScrollRect>(true);
            }
            // 2) 그래도 없으면 Scene 전체 검색 (NpcChatDemoUI가 패널 외부에 붙은 경우 대응)
            if (historyScrollRect == null)
            {
                var all = FindObjectsOfType<ScrollRect>(true);
                foreach (var sr in all)
                {
                    if (sr.name.StartsWith("History", System.StringComparison.OrdinalIgnoreCase)
                        || (sr.content != null && sr.content.name.StartsWith("History", System.StringComparison.OrdinalIgnoreCase)))
                    {
                        historyScrollRect = sr;
                        break;
                    }
                }
                if (historyScrollRect == null && all.Length > 0) historyScrollRect = all[0];
            }

            // 3) historyLogText 자동 검색
            if (historyLogText == null && historyScrollRect != null)
            {
                var content = historyScrollRect.content;
                if (content != null)
                    historyLogText = content.GetComponentInChildren<TMP_Text>(true);
            }
            if (historyLogText == null)
            {
                // Scene 전체에서 이름이 "History"로 시작하는 TMP_Text 검색
                foreach (var t in FindObjectsOfType<TMP_Text>(true))
                {
                    if (t.name.StartsWith("History", System.StringComparison.OrdinalIgnoreCase))
                    {
                        historyLogText = t;
                        break;
                    }
                }
            }

            // 진단 로그
            Debug.Log($"[NpcChatDemoUI] AutoWire — historyScrollRect={(historyScrollRect != null ? historyScrollRect.name : "null")}, " +
                      $"historyLogText={(historyLogText != null ? historyLogText.name : "null")}");
            if (historyLogText == null)
                Debug.LogWarning("[NpcChatDemoUI] historyLogText 슬롯 자동 연결 실패 — Inspector에서 직접 연결 필요");

            // 부모 체인 활성화 + 가시성 강제 (디버그용)
            if (historyLogText != null)
            {
                // 부모 체인 SetActive 확인
                Transform t = historyLogText.transform;
                while (t != null)
                {
                    if (!t.gameObject.activeSelf)
                    {
                        Debug.LogWarning($"[NpcChatDemoUI] '{t.name}' 비활성 — 활성화함");
                        t.gameObject.SetActive(true);
                    }
                    t = t.parent;
                }
                // 가시성
                historyLogText.color = new Color(1f, 1f, 1f, 1f);
                if (historyLogText.fontSize < 8) historyLogText.fontSize = 14;
                historyLogText.richText = true;
                historyLogText.enableWordWrapping = true;

                // ScrollRect의 Mask가 가리는 문제 회피: HistoryLogText를 ScrollRect 밖으로 reparent.
                // HistoryScroll과 같은 위치 (DialoguePanel 자식)로 옮김.
                if (historyScrollRect != null && historyLogText.transform.IsChildOf(historyScrollRect.transform))
                {
                    var scrollRt = historyScrollRect.GetComponent<RectTransform>();
                    var panelParent = scrollRt.parent;  // DialoguePanel
                    var rt = historyLogText.rectTransform;
                    rt.SetParent(panelParent, false);
                    // HistoryScroll과 동일 영역 anchor 복제
                    rt.anchorMin = scrollRt.anchorMin;
                    rt.anchorMax = scrollRt.anchorMax;
                    rt.pivot = new Vector2(0.5f, 1f);
                    rt.anchoredPosition = new Vector2(0, 0);
                    rt.offsetMin = scrollRt.offsetMin;
                    rt.offsetMax = scrollRt.offsetMax;
                    // 위쪽 시작
                    var off = rt.offsetMax;
                    rt.offsetMax = new Vector2(off.x, -8f);

                    // ContentSizeFitter 제거 (수동 height 관리)
                    var oldFitter = historyLogText.GetComponent<UnityEngine.UI.ContentSizeFitter>();
                    if (oldFitter != null) Destroy(oldFitter);

                    Debug.Log($"[NpcChatDemoUI] HistoryLogText를 ScrollRect 밖으로 reparent — 패널 자식으로 이동");
                    // ScrollRect는 더 이상 사용 안 함 (배경 BG만 남음)
                }
            }
        }

        private void Start()
        {
            UpdateDayText();
            ResetCurrentDisplay();

            if (autoConnectViaDialogueManager)
            {
                if (hideDropdownInAutoMode && npcDropdown != null)
                    npcDropdown.gameObject.SetActive(false);
                TrySubscribeDialogueManager();
                SetStatus("NPC와 가까이 가서 F 키를 누르세요");
            }
            else
            {
                _ = ConnectToCurrent();
            }
        }

        private void OnEnable() { TrySubscribeDialogueManager(); }
        private void OnDisable() { UnsubscribeDialogueManager(); }

        private void Update()
        {
            _client?.DispatchQueue();

            // InputField 자동 재포커스: 패널 빈 영역 클릭으로 포커스를 잃어도
            // 즉시 다시 활성. 다른 UI(드롭다운 옵션 등)가 선택됐을 때는 양보.
            if (inputField != null
                && inputField.gameObject.activeInHierarchy
                && inputField.interactable
                && !inputField.isFocused
                && !_waitingResponse)
            {
                var sel = UnityEngine.EventSystems.EventSystem.current?.currentSelectedGameObject;
                bool otherSelectable = sel != null
                    && sel != inputField.gameObject
                    && sel.GetComponent<UnityEngine.UI.Selectable>() != null;
                if (!otherSelectable)
                    inputField.ActivateInputField();
            }
        }

        private async void OnApplicationQuit()
        {
            if (_client != null) await _client.CloseAsync();
        }

        // ========== DialogueManager 연동 ==========
        private void TrySubscribeDialogueManager()
        {
            if (_subscribed || !autoConnectViaDialogueManager) return;
            if (DialogueManager.Instance == null) return;
            DialogueManager.Instance.OnDialogueStarted += HandleDialogueStarted;
            DialogueManager.Instance.OnDialogueEnded += HandleDialogueEnded;
            _subscribed = true;
        }

        private void UnsubscribeDialogueManager()
        {
            if (!_subscribed) return;
            if (DialogueManager.Instance != null)
            {
                DialogueManager.Instance.OnDialogueStarted -= HandleDialogueStarted;
                DialogueManager.Instance.OnDialogueEnded -= HandleDialogueEnded;
            }
            _subscribed = false;
        }

        private async void HandleDialogueStarted(NpcInteractor npc)
        {
            ClearHistoryLog();
            // npcName이 비어있으면 GameObject 이름 사용 (안전망)
            string rawName = string.IsNullOrEmpty(npc.npcName) ? npc.gameObject.name : npc.npcName;
            string displayName = DisplayName(rawName);
            ResetCurrentDisplay(displayName);
            HideQuestCard();
            SyncDropdown(rawName);

            SetStatus($"{displayName} 연결 중...");
            await ConnectToNpc(rawName);
            if (inputField != null) inputField.ActivateInputField();
        }

        private async void HandleDialogueEnded()
        {
            if (_client != null)
            {
                await _client.CloseAsync();
                _client = null;
            }
            SetStatus("NPC와 가까이 가서 F 키를 누르세요");
            ResetCurrentDisplay();
            HideQuestCard();
        }

        // ========== 연결 ==========
        private async Task ConnectToCurrent()
        {
            if (npcDropdown == null || npcDropdown.options.Count == 0) return;
            string npc = npcDropdown.options[npcDropdown.value].text;
            await ConnectToNpc(npc);
        }

        private async Task ConnectToNpc(string npc)
        {
            if (_client != null) await _client.CloseAsync();

            _client = new NpcChatClient(npc)
            {
                ServerHost = serverHost,
                ServerPort = serverPort,
            };
            _client.OnReady += HandleReady;
            _client.OnResponse += HandleResponse;
            _client.OnTickEvents += HandleTickEvents;
            _client.OnError += HandleError;
            _client.OnClosed += HandleClosed;

            await _client.ConnectAsync();
        }

        private void OnNpcChanged(int idx) { _ = ConnectToCurrent(); }

        // ========== 입력 ==========
        private void OnSendClicked()
        {
            if (_waitingResponse) return;
            if (_client == null || !_client.IsOpen)
            {
                SetStatus("연결되지 않음");
                return;
            }
            if (inputField == null) return;
            string text = inputField.text.Trim();
            if (string.IsNullOrEmpty(text)) return;

            inputField.text = "";
            inputField.interactable = false;
            _waitingResponse = true;

            AppendHistory($"<color=#{ColorToHex(playerColor)}><b>[Player]</b></color> {text}");
            SetStatus("응답 생성 중...");

            // 현재 응답 영역도 갱신 (대기 표시)
            if (currentResponseText != null) currentResponseText.text = "...";

            _ = _client.SendChatAsync(text);
        }

        private void OnTimeClicked()
        {
            if (_client == null || !_client.IsOpen) { SetStatus("연결되지 않음"); return; }
            SetStatus("시간 진행 중...");
            _ = _client.SendTimeAdvanceAsync();
        }

        // ========== ChatClient 이벤트 ==========
        private void HandleReady()
        {
            SetStatus($"{DisplayName(_client.Npc)} 연결됨 — 무엇을 물어보시겠어요?");
            if (inputField != null) inputField.interactable = true;
        }

        private void HandleResponse(ServerMessage resp)
        {
            _waitingResponse = false;
            string npcName = DisplayName(string.IsNullOrEmpty(resp.npc) ? "NPC" : resp.npc);
            Color color = GetNpcColor(npcName);
            string hex = ColorToHex(color);

            // 현재 응답 영역
            SetCurrentResponse(npcName, resp.text, color);

            // History 누적
            AppendHistory($"<color=#{hex}><b>[{npcName}]</b></color> {resp.text}");

            // 회상 메모리
            UpdateMemoryHint(resp.memories_used);
            if (showRetrievedMemories && resp.memories_used != null)
            {
                foreach (var m in resp.memories_used)
                {
                    AppendHistory(
                        $"<size=80%><color=#888>  ↳ [{m.source}·imp{m.importance}] {Truncate(m.text, 70)}</color></size>"
                    );
                }
            }

            // Quest
            ShowQuest(resp.quest);

            // Trust (친밀도)
            if (resp.trust > 0 || !string.IsNullOrEmpty(resp.trust_label))
            {
                string deltaStr = resp.trust_delta > 0
                    ? $"<color=#7fff7f>+{resp.trust_delta}</color>"
                    : (resp.trust_delta < 0 ? $"<color=#ff7f7f>{resp.trust_delta}</color>" : "");
                if (!string.IsNullOrEmpty(deltaStr))
                {
                    AppendHistory(
                        $"<size=80%><color=#888>  ↳ 친밀도 {deltaStr} → {resp.trust}/100 ({resp.trust_label})</color></size>"
                    );
                }
            }

            string trustStatus = (resp.trust > 0)
                ? $" · 친밀도 {resp.trust}/100 {resp.trust_label}"
                : "";
            SetStatus($"{npcName} 응답 완료 ({resp.latency_ms}ms){trustStatus}");

            if (inputField != null)
            {
                inputField.interactable = true;
                inputField.ActivateInputField();
            }
        }

        private void HandleTickEvents(ServerMessage tick)
        {
            _currentDay = tick.day;
            UpdateDayText();

            int n = tick.events != null ? tick.events.Length : 0;
            AppendHistory($"<size=90%><color=#ffb74d>━━━ Day {tick.day} ({n}개 정보 전달) ━━━</color></size>");

            if (showTickEvents && tick.events != null)
            {
                int max = Mathf.Min(tick.events.Length, 8);
                for (int i = 0; i < max; i++)
                {
                    var ev = tick.events[i];
                    Color cf = GetNpcColor(ev.from);
                    Color ct = GetNpcColor(ev.to);
                    AppendHistory(
                        $"<size=80%><color=#{ColorToHex(cf)}>{ev.from}</color>" +
                        $" <color=#888>→</color> " +
                        $"<color=#{ColorToHex(ct)}>{ev.to}</color>" +
                        $": <color=#aaa>\"{Truncate(ev.transformed, 50)}\"</color></size>"
                    );
                }
                if (tick.events.Length > max)
                    AppendHistory($"<size=80%><color=#888>  (외 {tick.events.Length - max}건)</color></size>");
            }

            // NPC-NPC 자율 대화 (Phase 2)
            if (tick.turns != null && tick.turns.Length > 0
                && !string.IsNullOrEmpty(tick.npc_a) && !string.IsNullOrEmpty(tick.npc_b))
            {
                Color ca = GetNpcColor(tick.npc_a);
                Color cb = GetNpcColor(tick.npc_b);
                AppendHistory(
                    $"<size=85%><color=#bcaaa4>━ <color=#{ColorToHex(ca)}>{tick.npc_a}</color>" +
                    $" ↔ <color=#{ColorToHex(cb)}>{tick.npc_b}</color> 자율 대화 ━</color></size>"
                );
                int convMax = Mathf.Min(tick.turns.Length, 6);
                for (int i = 0; i < convMax; i++)
                {
                    var t = tick.turns[i];
                    Color cs = GetNpcColor(t.speaker);
                    string name = string.IsNullOrEmpty(t.speaker_ko) ? t.speaker : t.speaker_ko;
                    AppendHistory(
                        $"<size=80%>  <color=#{ColorToHex(cs)}>{name}</color>" +
                        $": <color=#ccc>{Truncate(t.text, 80)}</color></size>"
                    );
                }
                if (tick.memory_saved)
                    AppendHistory($"<size=75%><color=#888>  → 양쪽 NPC 메모리에 기억됨</color></size>");
            }

            SetStatus($"Day {tick.day} 완료 ({n}개 전달)");
        }

        private void HandleError(string msg)
        {
            _waitingResponse = false;
            SetStatus($"에러: {msg}");
            if (inputField != null) inputField.interactable = true;
        }

        private void HandleClosed()
        {
            SetStatus("연결 종료");
        }

        // ========== UI 헬퍼 ==========
        /// <summary>
        /// 표시용 이름: "elias" → "Elias", "Hermann" → "Hermann"(그대로), 한글은 그대로.
        /// </summary>
        public static string DisplayName(string raw)
        {
            if (string.IsNullOrEmpty(raw)) return raw;
            // 한글이나 이미 대문자면 그대로
            if (!char.IsLetter(raw[0])) return raw;
            if (char.IsUpper(raw[0])) return raw;
            return char.ToUpper(raw[0]) + raw.Substring(1);
        }

        private void SetCurrentResponse(string npcName, string text, Color color)
        {
            if (currentNpcNameText != null)
            {
                currentNpcNameText.text = npcName;
                currentNpcNameText.color = color;
            }
            if (currentResponseText != null)
            {
                if (useTypewriterEffect)
                {
                    if (_typewriterRoutine != null) StopCoroutine(_typewriterRoutine);
                    _typewriterRoutine = StartCoroutine(TypewriterCoroutine(currentResponseText, text));
                }
                else
                {
                    currentResponseText.text = text;
                }
            }
        }

        private IEnumerator TypewriterCoroutine(TMP_Text target, string text)
        {
            target.text = "";
            var wait = new WaitForSeconds(typewriterCharDelay);
            for (int i = 0; i < text.Length; i++)
            {
                target.text += text[i];
                yield return wait;
            }
        }

        private void ResetCurrentDisplay(string npcName = null)
        {
            if (currentNpcNameText != null)
            {
                currentNpcNameText.text = string.IsNullOrEmpty(npcName) ? "" : npcName;
                if (!string.IsNullOrEmpty(npcName))
                    currentNpcNameText.color = GetNpcColor(npcName);
            }
            if (currentResponseText != null) currentResponseText.text = "";
            if (memoryHintText != null) memoryHintText.text = "";
        }

        private void UpdateMemoryHint(Memory[] mems)
        {
            if (memoryHintText == null) return;
            if (mems == null || mems.Length == 0) { memoryHintText.text = ""; return; }
            var first = mems[0];
            memoryHintText.text = $"↳ {first.source}: {Truncate(first.text, 60)}";
        }

        private void ShowQuest(Quest q)
        {
            if (questCard == null) return;
            if (q == null || !q.IsValid) { questCard.SetActive(false); return; }

            questCard.SetActive(true);
            if (questTitleText != null) questTitleText.text = $"★ {q.title}";
            if (questDescText != null) questDescText.text = q.description;
            if (questRewardText != null)
                questRewardText.text = string.IsNullOrEmpty(q.reward) ? "" : $"보상: {q.reward}";

            // history에도 표시
            AppendHistory($"<color=#ffd54f>★ Quest: {q.title}</color>");
            AppendHistory($"<size=85%><color=#bbb>  {q.description}</color></size>");
            if (!string.IsNullOrEmpty(q.reward))
                AppendHistory($"<size=85%><color=#bbb>  보상: {q.reward}</color></size>");
        }

        private void HideQuestCard()
        {
            if (questCard != null) questCard.SetActive(false);
        }

        private void AppendHistory(string line)
        {
            if (!keepHistoryLog) return;
            _log.AppendLine(line);
            if (historyLogText != null)
            {
                historyLogText.text = _log.ToString();
                historyLogText.ForceMeshUpdate();

                // ContentSizeFitter 없이 직접 height 설정
                float preferredHeight = historyLogText.preferredHeight + 16f;
                var textRt = historyLogText.rectTransform;
                textRt.sizeDelta = new Vector2(textRt.sizeDelta.x, preferredHeight);
                if (historyScrollRect != null && historyScrollRect.content != null)
                {
                    var c = historyScrollRect.content;
                    c.sizeDelta = new Vector2(c.sizeDelta.x, preferredHeight);
                }

                // 진단: width / 부모 alpha / 위치 확인
                var rect = textRt.rect;
                var worldPos = textRt.position;
                var canvasGroup = GetComponentInParent<CanvasGroup>();
                float parentAlpha = canvasGroup != null ? canvasGroup.alpha : 1f;
                Debug.Log($"[AppendHistory v3] rect=(w={rect.width:F1},h={rect.height:F1}), worldPos=({worldPos.x:F0},{worldPos.y:F0}), fontSize={historyLogText.fontSize}, color={historyLogText.color}, parentAlpha={parentAlpha:F2}");

                // 자동 스크롤
                if (historyScrollRect != null)
                {
                    Canvas.ForceUpdateCanvases();
                    historyScrollRect.verticalNormalizedPosition = 0f;
                }
            }
            else
            {
                Debug.LogWarning("[NpcChatDemoUI] AppendHistory: historyLogText가 null — 메시지 누락");
            }
        }

        private void ClearHistoryLog()
        {
            _log.Clear();
            if (historyLogText != null) historyLogText.text = "";
        }

        private void SyncDropdown(string npcName)
        {
            if (npcDropdown == null || !npcDropdown.gameObject.activeSelf) return;
            for (int i = 0; i < npcDropdown.options.Count; i++)
            {
                if (npcDropdown.options[i].text == npcName)
                {
                    npcDropdown.SetValueWithoutNotify(i);
                    break;
                }
            }
        }

        private void UpdateDayText()
        {
            if (dayText != null) dayText.text = $"Day {_currentDay}";
        }

        private void SetStatus(string s)
        {
            if (statusText != null) statusText.text = s;
        }

        private void BuildColorMap()
        {
            _colorMap = new Dictionary<string, Color>();
            if (npcColors == null) return;
            foreach (var e in npcColors)
                if (!string.IsNullOrEmpty(e.npc)) _colorMap[e.npc] = e.color;
        }

        private Color GetNpcColor(string npc)
        {
            if (_colorMap == null) BuildColorMap();
            return (_colorMap != null && _colorMap.TryGetValue(npc, out var c)) ? c : Color.white;
        }

        private static string ColorToHex(Color c)
        {
            return $"{Mathf.RoundToInt(c.r * 255):X2}{Mathf.RoundToInt(c.g * 255):X2}{Mathf.RoundToInt(c.b * 255):X2}";
        }

        private static string Truncate(string s, int n) => s.Length <= n ? s : s.Substring(0, n) + "…";
    }
}
