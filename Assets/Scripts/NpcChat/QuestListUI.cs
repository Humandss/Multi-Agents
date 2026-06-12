using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace NpcChat
{
    /// <summary>
    /// 좌측 퀘스트 리스트 패널 + 시작 확인 모달.
    ///
    /// 흐름:
    ///   대화 시작 → 현재 NPC의 퀘스트 fetch + 패널 표시
    ///   [시작 가능] 엔트리 클릭 → "「제목」 시작하시겠습니까? 예/아니오" 모달
    ///   예 → NpcChatDemoUI.ProposeQuest (대화창 구분선 + NPC 퀘스트 대사)
    ///   수락/거절은 채팅에서 플레이어가 직접 답 → 서버 분류 → 리스트 자동 갱신
    ///   [진행 중] 엔트리 → [완료] 버튼 (trust +10)
    ///
    /// 이 컴포넌트는 항상 active한 루트에 붙고, panelRoot(자식)만 켜고 끔.
    /// </summary>
    public class QuestListUI : MonoBehaviour
    {
        [Header("패널 슬롯")]
        public GameObject panelRoot;
        public Transform contentRoot;
        public TMP_Text emptyLabel;
        public TMP_FontAsset koreanFont;     // 런타임 생성 텍스트용

        [Header("확인 모달 슬롯")]
        public GameObject modalRoot;
        public TMP_Text modalTitle;
        public Button modalYesBtn;
        public Button modalNoBtn;

        [Header("동작")]
        public bool showWithDialogue = true;

        static readonly Dictionary<string, Color> NPC_COLOR = new Dictionary<string, Color>
        {
            { "elias",     new Color(0.4f, 0.85f, 0.95f) },
            { "hermann",   new Color(0.95f, 0.45f, 0.45f) },
            { "mathilda",  new Color(0.98f, 0.85f, 0.4f) },
            { "finn",      new Color(0.9f, 0.55f, 0.95f) },
            { "bernhardt", new Color(0.5f, 0.9f, 0.55f) },
        };

        readonly List<GameObject> _spawned = new List<GameObject>();
        QuestEntry _modalTarget;
        NpcChatDemoUI _demoUI;
        System.Action<NpcInteractor> _onDialogueStarted;
        System.Action _onDialogueEnded;

        void Start()
        {
            _demoUI = FindObjectOfType<NpcChatDemoUI>(true);

            if (QuestManager.Instance != null)
                QuestManager.Instance.OnQuestsChanged += Rebuild;

            if (showWithDialogue && DialogueManager.Instance != null)
            {
                _onDialogueStarted = npc =>
                {
                    string raw = string.IsNullOrEmpty(npc.npcName)
                        ? npc.gameObject.name : npc.npcName;
                    if (QuestManager.Instance != null)
                        QuestManager.Instance.FetchQuestsFor(raw);
                    Show(true);
                };
                _onDialogueEnded = () =>
                {
                    CloseModal();
                    Show(false);
                };
                DialogueManager.Instance.OnDialogueStarted += _onDialogueStarted;
                DialogueManager.Instance.OnDialogueEnded += _onDialogueEnded;
            }

            if (modalYesBtn != null) modalYesBtn.onClick.AddListener(OnModalYes);
            if (modalNoBtn != null) modalNoBtn.onClick.AddListener(CloseModal);

            CloseModal();
            Show(!showWithDialogue);
            Rebuild();
        }

        void OnDestroy()
        {
            if (QuestManager.Instance != null)
                QuestManager.Instance.OnQuestsChanged -= Rebuild;
            if (DialogueManager.Instance != null)
            {
                if (_onDialogueStarted != null)
                    DialogueManager.Instance.OnDialogueStarted -= _onDialogueStarted;
                if (_onDialogueEnded != null)
                    DialogueManager.Instance.OnDialogueEnded -= _onDialogueEnded;
            }
        }

        public void Show(bool on)
        {
            if (panelRoot != null) panelRoot.SetActive(on);
        }

        // ========== 확인 모달 ==========
        void OpenConfirm(QuestEntry e)
        {
            _modalTarget = e;
            if (modalTitle != null)
                modalTitle.text = $"「{e.title}」\n시작하시겠습니까?";
            if (modalRoot != null) modalRoot.SetActive(true);
        }

        void OnModalYes()
        {
            var target = _modalTarget;
            CloseModal();
            if (target == null) return;
            if (_demoUI == null) _demoUI = FindObjectOfType<NpcChatDemoUI>(true);
            if (_demoUI != null) _demoUI.ProposeQuest(target);
        }

        void CloseModal()
        {
            _modalTarget = null;
            if (modalRoot != null) modalRoot.SetActive(false);
        }

        // ========== 리스트 구성 ==========
        void Rebuild()
        {
            foreach (var go in _spawned)
                if (go != null) Destroy(go);
            _spawned.Clear();

            var quests = QuestManager.Instance != null
                ? QuestManager.Instance.quests
                : new List<QuestEntry>();

            if (emptyLabel != null)
                emptyLabel.gameObject.SetActive(quests.Count == 0);
            if (quests.Count == 0) return;

            // 상태별 섹션: 진행 중 → 시작 가능 → 완료
            var accepted = quests.FindAll(q => q.state == QuestState.Accepted);
            var available = quests.FindAll(q => q.state == QuestState.Available);
            var completed = quests.FindAll(q => q.state == QuestState.Completed);

            AddSection("진행 중", new Color(0.5f, 0.77f, 1f), accepted);
            AddSection("시작 가능", new Color(1f, 0.85f, 0.4f), available);
            AddSection("완료", new Color(0.5f, 0.85f, 0.5f), completed);
        }

        void AddSection(string title, Color color, List<QuestEntry> entries)
        {
            if (entries.Count == 0) return;
            _spawned.Add(CreateSectionHeader($"{title} ({entries.Count})", color));
            foreach (var e in entries)
                _spawned.Add(CreateEntry(e));
        }

        GameObject CreateSectionHeader(string text, Color color)
        {
            var go = new GameObject($"Section_{text}",
                typeof(RectTransform), typeof(LayoutElement));
            go.transform.SetParent(contentRoot, false);
            var le = go.GetComponent<LayoutElement>();
            le.minHeight = 26;
            le.preferredHeight = 26;

            var label = NewText(go.transform, $"■ {text}", 13, color, FontStyles.Bold);
            SetRect(label, Vector2.zero, Vector2.one,
                new Vector2(4, 2), new Vector2(-4, -2));
            return go;
        }

        GameObject CreateEntry(QuestEntry e)
        {
            bool done = e.state == QuestState.Completed;
            bool accepted = e.state == QuestState.Accepted;
            bool clickable = e.state == QuestState.Available && e.eligible;
            bool locked = e.state == QuestState.Available && !e.eligible;

            var go = new GameObject($"Quest_{e.title}",
                typeof(RectTransform), typeof(Image), typeof(LayoutElement));
            go.transform.SetParent(contentRoot, false);
            var bg = go.GetComponent<Image>();
            bg.color = done ? new Color(0.09f, 0.11f, 0.10f, 0.85f)
                : accepted ? new Color(0.12f, 0.15f, 0.20f, 0.95f)
                : locked ? new Color(0.10f, 0.10f, 0.11f, 0.7f)
                : new Color(0.11f, 0.13f, 0.17f, 0.95f);
            var le = go.GetComponent<LayoutElement>();
            le.minHeight = 112;
            le.preferredHeight = 112;

            string giverKey = (e.giver ?? "").ToLowerInvariant();
            Color npcColor = NPC_COLOR.TryGetValue(giverKey, out var c)
                ? c : new Color(1f, 0.85f, 0.4f);

            // 제목 + 상태 마크
            string mark = done ? "<color=#7fff7f>[완료]</color> "
                : accepted ? "<color=#7fc4ff>[진행 중]</color> "
                : "★ ";
            Color titleColor = done || locked
                ? new Color(0.55f, 0.58f, 0.55f) : npcColor;
            var title = NewText(go.transform, $"{mark}{e.title}", 15, titleColor, FontStyles.Bold);
            SetRect(title, new Vector2(0, 1), new Vector2(1, 1),
                new Vector2(10, -30), new Vector2(-10, -4));

            // 설명 — 템플릿 형식 "목표: ..."
            var desc = NewText(go.transform, $"목표: {e.description}", 12,
                locked ? new Color(0.5f, 0.5f, 0.52f) : new Color(0.82f, 0.84f, 0.86f),
                FontStyles.Normal);
            desc.enableWordWrapping = true;
            SetRect(desc, new Vector2(0, 1), new Vector2(1, 1),
                new Vector2(10, -72), new Vector2(-10, -30));

            // 좌하단: 보상 또는 잠금 안내
            string footLeft = locked
                ? $"친밀도 {e.trustRequired} 필요"
                : (string.IsNullOrEmpty(e.reward) ? "" : $"보상: {e.reward}");
            var foot = NewText(go.transform, footLeft, 11,
                locked ? new Color(0.7f, 0.5f, 0.4f) : new Color(0.65f, 0.62f, 0.5f),
                FontStyles.Normal);
            SetRect(foot, new Vector2(0, 0), new Vector2(0.6f, 0),
                new Vector2(10, 8), new Vector2(0, 34));

            // 우하단: 상태별 액션
            if (clickable)
            {
                // 엔트리 전체 클릭 → 시작 확인 모달
                var btn = go.AddComponent<Button>();
                var captured = e;
                btn.onClick.AddListener(() => OpenConfirm(captured));

                var hint = NewText(go.transform, "클릭하여 시작", 11,
                    new Color(0.55f, 0.75f, 1f), FontStyles.Italic);
                hint.alignment = TextAlignmentOptions.Right;
                SetRect(hint, new Vector2(0.5f, 0), new Vector2(1, 0),
                    new Vector2(0, 8), new Vector2(-10, 34));
            }
            else if (accepted)
            {
                var btnGo = new GameObject("CompleteBtn",
                    typeof(RectTransform), typeof(Image), typeof(Button));
                btnGo.transform.SetParent(go.transform, false);
                btnGo.GetComponent<Image>().color = new Color(0.2f, 0.45f, 0.8f, 1f);
                var btnRt = btnGo.GetComponent<RectTransform>();
                btnRt.anchorMin = new Vector2(1, 0);
                btnRt.anchorMax = new Vector2(1, 0);
                btnRt.pivot = new Vector2(1, 0);
                btnRt.anchoredPosition = new Vector2(-8, 8);
                btnRt.sizeDelta = new Vector2(64, 26);

                var btnLabel = NewText(btnGo.transform, "완료", 12, Color.white, FontStyles.Bold);
                btnLabel.alignment = TextAlignmentOptions.Center;
                SetRect(btnLabel, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);

                var captured = e;
                btnGo.GetComponent<Button>().onClick.AddListener(() =>
                {
                    if (QuestManager.Instance != null)
                        QuestManager.Instance.CompleteQuest(captured);
                });
            }
            else if (done)
            {
                var doneLabel = NewText(go.transform, "완료됨 · 친밀도 +10", 11,
                    new Color(0.5f, 0.85f, 0.5f), FontStyles.Normal);
                doneLabel.alignment = TextAlignmentOptions.Right;
                SetRect(doneLabel, new Vector2(0.5f, 0), new Vector2(1, 0),
                    new Vector2(0, 8), new Vector2(-10, 34));
            }

            return go;
        }

        // ========== 헬퍼 ==========
        TMP_Text NewText(Transform parent, string text, float size, Color color, FontStyles style)
        {
            var go = new GameObject("Text", typeof(RectTransform));
            go.transform.SetParent(parent, false);
            var tmp = go.AddComponent<TextMeshProUGUI>();
            tmp.text = text;
            tmp.fontSize = size;
            tmp.color = color;
            tmp.fontStyle = style;
            tmp.richText = true;
            tmp.alignment = TextAlignmentOptions.TopLeft;
            tmp.raycastTarget = false;   // 엔트리 Button 클릭 방해 X
            if (koreanFont != null) tmp.font = koreanFont;
            return tmp;
        }

        static void SetRect(Component c, Vector2 anchorMin, Vector2 anchorMax,
                            Vector2 offsetMin, Vector2 offsetMax)
        {
            var rt = c.GetComponent<RectTransform>();
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            rt.offsetMin = offsetMin;
            rt.offsetMax = offsetMax;
        }
    }
}
