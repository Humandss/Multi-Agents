#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using TMPro;

namespace NpcChat.EditorTools
{
    /// <summary>
    /// 메뉴 한 번 클릭으로 멋진 대화 UI 자동 생성 + NpcChatDemoUI 슬롯 자동 연결.
    ///
    /// Unity 메뉴: Tools > NpcChat > Create Dialogue UI Panel
    ///
    /// 생성되는 것:
    ///   Canvas (없으면 새로)
    ///     ├─ EventSystem
    ///     └─ DialoguePanel (오른쪽 50%)
    ///         ├─ Background (반투명 dark)
    ///         ├─ Header (NPC 이름 + 상태)
    ///         ├─ MainResponseArea (큰 응답 영역)
    ///         ├─ MemoryHint (회상 메모리)
    ///         ├─ QuestCard (Quest 표시)
    ///         ├─ HistoryScroll (대화 로그)
    ///         └─ InputBar (InputField + SendButton)
    /// </summary>
    public static class DialogueUIBuilder
    {
        // ============ 색상 팔레트 ============
        static readonly Color BgDark          = new Color(0.07f, 0.08f, 0.10f, 0.92f);
        static readonly Color BgPanel         = new Color(0.12f, 0.13f, 0.16f, 0.95f);
        static readonly Color BgAccent        = new Color(0.18f, 0.20f, 0.25f, 1f);
        static readonly Color BgInput         = new Color(0.05f, 0.06f, 0.08f, 1f);
        static readonly Color TextPrimary     = new Color(0.95f, 0.95f, 0.95f, 1f);
        static readonly Color TextSecondary   = new Color(0.70f, 0.72f, 0.75f, 1f);
        static readonly Color TextMuted       = new Color(0.55f, 0.55f, 0.55f, 1f);
        static readonly Color AccentYellow    = new Color(1f, 0.85f, 0.35f, 1f);
        static readonly Color AccentNpcName   = new Color(1f, 0.90f, 0.55f, 1f);
        static readonly Color QuestGold       = new Color(1f, 0.83f, 0.30f, 1f);
        static readonly Color BorderAccent    = new Color(0.45f, 0.50f, 0.65f, 0.7f);

        [MenuItem("Tools/NpcChat/Create Dialogue UI Panel", priority = 1)]
        public static void CreateDialogueUI()
        {
            EnsureEventSystem();
            var canvas = FindOrCreateCanvas();
            var panel = BuildPanel(canvas);

            // 생성된 UI에 한국어 폰트 자동 적용
            var koreanFont = FindKoreanFontAsset();
            if (koreanFont != null)
            {
                ApplyFontToAllTMP(panel, koreanFont);
                Debug.Log($"[DialogueUIBuilder] 한국어 폰트 적용: {koreanFont.name}");
            }
            else
            {
                Debug.LogWarning("[DialogueUIBuilder] Pretendard/NotoSans/NanumGothic TMP Font Asset 못 찾음. " +
                    "메뉴 Tools > NpcChat > Apply Korean Font to Scene 으로 수동 적용 가능.");
            }

            EditorSceneManager.MarkSceneDirty(panel.scene);
            Selection.activeGameObject = panel;
            EditorGUIUtility.PingObject(panel);

            Debug.Log("[DialogueUIBuilder] DialoguePanel 생성 완료. " +
                      "NpcChatDemoUI 슬롯 자동 연결됨. " +
                      "UIController/DialogueManager의 'Dialogue Panel' 슬롯에 이 패널 연결하세요.");
        }

        [MenuItem("Tools/NpcChat/Create Interaction Prompt", priority = 2)]
        public static void CreateInteractionPrompt()
        {
            EnsureEventSystem();
            var canvas = FindOrCreateCanvas();
            var go = BuildInteractionPrompt(canvas);

            var koreanFont = FindKoreanFontAsset();
            if (koreanFont != null) ApplyFontToAllTMP(go, koreanFont);

            EditorSceneManager.MarkSceneDirty(go.scene);
            Selection.activeGameObject = go;
            EditorGUIUtility.PingObject(go);
            Debug.Log("[DialogueUIBuilder] InteractionPrompt 생성 완료. " +
                      "NpcInteractor가 자동으로 prompt 호출함.");
        }

        [MenuItem("Tools/NpcChat/Create Town Name Display", priority = 3)]
        public static void CreateTownNameDisplay()
        {
            EnsureEventSystem();
            var canvas = FindOrCreateCanvas();
            var go = BuildTownNameDisplay(canvas);

            // 한국어 폰트 자동 적용
            var koreanFont = FindKoreanFontAsset();
            if (koreanFont != null) ApplyFontToAllTMP(go, koreanFont);

            EditorSceneManager.MarkSceneDirty(go.scene);
            Selection.activeGameObject = go;
            EditorGUIUtility.PingObject(go);
            Debug.Log("[DialogueUIBuilder] TownNameDisplay 생성 완료. " +
                      "Inspector에서 townName/subtitle 편집 가능. Play 시 자동 재생.");
        }

        [MenuItem("Tools/NpcChat/Apply Korean Font to Scene", priority = 10)]
        public static void ApplyKoreanFontToScene()
        {
            var font = FindKoreanFontAsset();
            if (font == null)
            {
                EditorUtility.DisplayDialog("폰트 못 찾음",
                    "Pretendard / NotoSans / NanumGothic TMP Font Asset을 프로젝트에서 못 찾았어요.\n" +
                    "Window > TextMeshPro > Font Asset Creator로 먼저 만들어주세요.",
                    "확인");
                return;
            }

            int count = 0;
            foreach (var tmp in Object.FindObjectsOfType<TMP_Text>(true))
            {
                Undo.RecordObject(tmp, "Apply Korean Font");
                tmp.font = font;
                count++;
            }

            // TMP Settings의 default fallback에도 추가 (없으면)
            AddToTmpFallback(font);

            EditorSceneManager.MarkSceneDirty(UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene());
            Debug.Log($"[DialogueUIBuilder] Scene의 {count}개 TMP_Text에 '{font.name}' 적용. " +
                      "TMP Settings Fallback Font Assets에도 등록.");
        }

        // ============ Korean Font 자동 detect ============
        internal static TMP_FontAsset FindKoreanFontAsset()
        {
            // 우선순위: Pretendard > NotoSans > NanumGothic > Malgun > 기타 SDF
            string[] keywords = { "Pretendard", "NotoSans", "NanumGothic", "Nanum", "Malgun", "맑은" };
            foreach (var kw in keywords)
            {
                var guids = AssetDatabase.FindAssets($"{kw} t:TMP_FontAsset");
                if (guids.Length > 0)
                {
                    string path = AssetDatabase.GUIDToAssetPath(guids[0]);
                    return AssetDatabase.LoadAssetAtPath<TMP_FontAsset>(path);
                }
            }
            return null;
        }

        internal static void ApplyFontToAllTMP(GameObject root, TMP_FontAsset font)
        {
            var texts = root.GetComponentsInChildren<TMP_Text>(true);
            foreach (var t in texts) t.font = font;
        }

        static void AddToTmpFallback(TMP_FontAsset font)
        {
            var settings = TMP_Settings.instance;
            if (settings == null) return;
            var so = new SerializedObject(settings);
            var prop = so.FindProperty("m_fallbackFontAssets");
            if (prop == null || !prop.isArray) return;

            // 이미 있으면 스킵
            for (int i = 0; i < prop.arraySize; i++)
            {
                var el = prop.GetArrayElementAtIndex(i).objectReferenceValue;
                if (el == font) return;
            }
            prop.arraySize++;
            prop.GetArrayElementAtIndex(prop.arraySize - 1).objectReferenceValue = font;
            so.ApplyModifiedProperties();
        }

        // ============ Canvas / EventSystem ============
        internal static void EnsureEventSystem()
        {
            if (Object.FindObjectOfType<EventSystem>() == null)
            {
                var go = new GameObject("EventSystem",
                    typeof(EventSystem), typeof(StandaloneInputModule));
                Undo.RegisterCreatedObjectUndo(go, "Create EventSystem");
            }
        }

        internal static Canvas FindOrCreateCanvas()
        {
            var canvas = Object.FindObjectOfType<Canvas>();
            if (canvas != null) return canvas;

            var go = new GameObject("Canvas",
                typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            canvas = go.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;

            var scaler = go.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
            scaler.matchWidthOrHeight = 0.5f;

            Undo.RegisterCreatedObjectUndo(go, "Create Canvas");
            return canvas;
        }

        // ============ Main Panel ============
        static GameObject BuildPanel(Canvas canvas)
        {
            // 패널 root
            var panel = CreateUIObject("DialoguePanel", canvas.transform);
            var rt = panel.GetComponent<RectTransform>();
            // 오른쪽 절반 차지
            rt.anchorMin = new Vector2(0.50f, 0.05f);
            rt.anchorMax = new Vector2(0.98f, 0.95f);
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;

            // 배경
            var bg = panel.AddComponent<Image>();
            bg.color = BgDark;
            bg.raycastTarget = true;

            // 살짝 둥근 outline 느낌 — Outline 컴포넌트
            var outline = panel.AddComponent<Outline>();
            outline.effectColor = BorderAccent;
            outline.effectDistance = new Vector2(2, -2);

            // === Header ===
            var (npcNameText, statusText) = BuildHeader(panel.transform);

            // === Main Response ===
            var responseText = BuildMainResponse(panel.transform);

            // === Memory Hint ===
            var memoryHintText = BuildMemoryHint(panel.transform);

            // === Quest Card ===
            var quest = BuildQuestCard(panel.transform);

            // === History Scroll ===
            var (historyText, historyScroll) = BuildHistoryScroll(panel.transform);

            // === Input Bar ===
            var (inputField, sendButton) = BuildInputBar(panel.transform);

            // === NpcChatDemoUI 컴포넌트 부착 + 슬롯 연결 ===
            var demoUI = panel.AddComponent<NpcChatDemoUI>();
            demoUI.currentNpcNameText = npcNameText;
            demoUI.currentResponseText = responseText;
            demoUI.statusText = statusText;
            demoUI.memoryHintText = memoryHintText;
            demoUI.historyLogText = historyText;
            demoUI.historyScrollRect = historyScroll;
            demoUI.inputField = inputField;
            demoUI.sendButton = sendButton;
            demoUI.questCard = quest.card;
            demoUI.questTitleText = quest.titleText;
            demoUI.questDescText = quest.descText;
            demoUI.questRewardText = quest.rewardText;
            demoUI.autoConnectViaDialogueManager = true;
            demoUI.hideDropdownInAutoMode = true;

            Undo.RegisterCreatedObjectUndo(panel, "Create DialoguePanel");
            return panel;
        }

        // ============ Header ============
        static (TMP_Text name, TMP_Text status) BuildHeader(Transform parent)
        {
            var header = CreateUIObject("Header", parent);
            SetRectFull(header, 0f, 0.88f, 1f, 1f, 12, 12, 12, 8);

            var bg = header.AddComponent<Image>();
            bg.color = BgAccent;

            // 하단 줄 강조 (Image 자식으로 얇은 라인)
            var line = CreateUIObject("BottomLine", header.transform);
            var lineRt = line.GetComponent<RectTransform>();
            lineRt.anchorMin = new Vector2(0f, 0f);
            lineRt.anchorMax = new Vector2(1f, 0f);
            lineRt.sizeDelta = new Vector2(0, 2);
            lineRt.anchoredPosition = Vector2.zero;
            var lineImg = line.AddComponent<Image>();
            lineImg.color = AccentNpcName;

            // NPC 이름 (left)
            var nameGo = CreateUIObject("CurrentNpcNameText", header.transform);
            SetRectFull(nameGo, 0f, 0f, 0.65f, 1f, 16, 0, 8, 0);
            var nameText = nameGo.AddComponent<TextMeshProUGUI>();
            nameText.text = "NPC 이름";
            nameText.color = AccentNpcName;
            nameText.fontSize = 34;
            nameText.fontStyle = FontStyles.Bold;
            nameText.alignment = TextAlignmentOptions.Left;
            nameText.enableAutoSizing = false;

            // 상태 텍스트 (right)
            var statusGo = CreateUIObject("StatusText", header.transform);
            SetRectFull(statusGo, 0.55f, 0f, 1f, 1f, 0, 16, 0, 0);
            var statusText = statusGo.AddComponent<TextMeshProUGUI>();
            statusText.text = "NPC와 가까이 가서 F 키를 누르세요";
            statusText.color = TextMuted;
            statusText.fontSize = 14;
            statusText.alignment = TextAlignmentOptions.Right;
            statusText.enableWordWrapping = true;

            return (nameText, statusText);
        }

        // ============ Main Response ============
        static TMP_Text BuildMainResponse(Transform parent)
        {
            var box = CreateUIObject("MainResponseArea", parent);
            SetRectFull(box, 0f, 0.65f, 1f, 0.88f, 16, 16, 0, 8);

            var bg = box.AddComponent<Image>();
            bg.color = BgPanel;

            var textGo = CreateUIObject("CurrentResponseText", box.transform);
            SetRectFull(textGo, 0f, 0f, 1f, 1f, 16, 16, 16, 16);
            var text = textGo.AddComponent<TextMeshProUGUI>();
            text.text = "";
            text.color = TextPrimary;
            text.fontSize = 22;
            text.fontStyle = FontStyles.Normal;
            text.alignment = TextAlignmentOptions.TopLeft;
            text.enableWordWrapping = true;

            return text;
        }

        // ============ Memory Hint ============
        static TMP_Text BuildMemoryHint(Transform parent)
        {
            var go = CreateUIObject("MemoryHintText", parent);
            SetRectFull(go, 0f, 0.61f, 1f, 0.65f, 20, 16, 0, 0);
            var text = go.AddComponent<TextMeshProUGUI>();
            text.text = "";
            text.color = TextMuted;
            text.fontSize = 13;
            text.fontStyle = FontStyles.Italic;
            text.alignment = TextAlignmentOptions.Left;
            text.enableWordWrapping = false;
            text.overflowMode = TextOverflowModes.Ellipsis;
            return text;
        }

        // ============ Quest Card ============
        struct QuestRefs { public GameObject card; public TMP_Text titleText, descText, rewardText; }

        static QuestRefs BuildQuestCard(Transform parent)
        {
            var card = CreateUIObject("QuestCard", parent);
            SetRectFull(card, 0f, 0.48f, 1f, 0.60f, 16, 16, 0, 4);
            card.SetActive(false);

            var bg = card.AddComponent<Image>();
            bg.color = new Color(0.20f, 0.16f, 0.08f, 0.95f);
            var outline = card.AddComponent<Outline>();
            outline.effectColor = QuestGold;
            outline.effectDistance = new Vector2(2, -2);

            // Title
            var titleGo = CreateUIObject("QuestTitleText", card.transform);
            SetRectFull(titleGo, 0f, 0.55f, 1f, 1f, 14, 14, 8, 4);
            var titleText = titleGo.AddComponent<TextMeshProUGUI>();
            titleText.text = "★ Quest";
            titleText.color = QuestGold;
            titleText.fontSize = 18;
            titleText.fontStyle = FontStyles.Bold;
            titleText.alignment = TextAlignmentOptions.Left;

            // Description
            var descGo = CreateUIObject("QuestDescText", card.transform);
            SetRectFull(descGo, 0f, 0.20f, 1f, 0.55f, 14, 14, 0, 0);
            var descText = descGo.AddComponent<TextMeshProUGUI>();
            descText.text = "";
            descText.color = TextPrimary;
            descText.fontSize = 14;
            descText.alignment = TextAlignmentOptions.Left;
            descText.enableWordWrapping = true;

            // Reward
            var rewardGo = CreateUIObject("QuestRewardText", card.transform);
            SetRectFull(rewardGo, 0f, 0f, 1f, 0.20f, 14, 14, 0, 4);
            var rewardText = rewardGo.AddComponent<TextMeshProUGUI>();
            rewardText.text = "";
            rewardText.color = AccentYellow;
            rewardText.fontSize = 13;
            rewardText.alignment = TextAlignmentOptions.Left;

            return new QuestRefs { card = card, titleText = titleText, descText = descText, rewardText = rewardText };
        }

        // ============ History Scroll ============
        static (TMP_Text text, ScrollRect scroll) BuildHistoryScroll(Transform parent)
        {
            var scrollGo = CreateUIObject("HistoryScroll", parent);
            SetRectFull(scrollGo, 0f, 0.12f, 1f, 0.46f, 16, 16, 0, 4);
            var bg = scrollGo.AddComponent<Image>();
            bg.color = BgPanel;

            var scroll = scrollGo.AddComponent<ScrollRect>();
            scroll.horizontal = false;
            scroll.vertical = true;
            scroll.movementType = ScrollRect.MovementType.Clamped;
            scroll.scrollSensitivity = 30f;

            // Viewport
            var viewport = CreateUIObject("Viewport", scrollGo.transform);
            SetRectFull(viewport, 0f, 0f, 1f, 1f, 8, 8, 8, 8);
            var viewportImg = viewport.AddComponent<Image>();
            viewportImg.color = new Color(0, 0, 0, 0.001f); // 거의 투명
            var mask = viewport.AddComponent<Mask>();
            mask.showMaskGraphic = false;
            scroll.viewport = viewport.GetComponent<RectTransform>();

            // Content
            var content = CreateUIObject("Content", viewport.transform);
            var contentRt = content.GetComponent<RectTransform>();
            contentRt.anchorMin = new Vector2(0, 1);
            contentRt.anchorMax = new Vector2(1, 1);
            contentRt.pivot = new Vector2(0.5f, 1);
            contentRt.anchoredPosition = Vector2.zero;
            contentRt.sizeDelta = new Vector2(0, 0);
            scroll.content = contentRt;

            var fitter = content.AddComponent<ContentSizeFitter>();
            fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;

            // History Text
            var textGo = CreateUIObject("HistoryLogText", content.transform);
            SetRectFull(textGo, 0f, 0f, 1f, 1f, 8, 8, 8, 8);
            var text = textGo.AddComponent<TextMeshProUGUI>();
            text.text = "";
            text.color = TextSecondary;
            text.fontSize = 14;
            text.alignment = TextAlignmentOptions.TopLeft;
            text.enableWordWrapping = true;
            text.richText = true;

            return (text, scroll);
        }

        // ============ Input Bar ============
        static (TMP_InputField input, Button send) BuildInputBar(Transform parent)
        {
            var bar = CreateUIObject("InputBar", parent);
            SetRectFull(bar, 0f, 0f, 1f, 0.12f, 16, 16, 4, 12);

            // InputField (left)
            var inputGo = CreateUIObject("InputField", bar.transform);
            SetRectFull(inputGo, 0f, 0f, 0.78f, 1f, 0, 8, 0, 0);
            var inputBg = inputGo.AddComponent<Image>();
            inputBg.color = BgInput;
            var inputOutline = inputGo.AddComponent<Outline>();
            inputOutline.effectColor = BorderAccent;
            inputOutline.effectDistance = new Vector2(1, -1);

            // InputField text area
            var textArea = CreateUIObject("TextArea", inputGo.transform);
            SetRectFull(textArea, 0f, 0f, 1f, 1f, 12, 12, 8, 8);
            var rtMask = textArea.AddComponent<RectMask2D>();

            // placeholder
            var placeholderGo = CreateUIObject("Placeholder", textArea.transform);
            SetRectFull(placeholderGo, 0f, 0f, 1f, 1f, 0, 0, 0, 0);
            var placeholderText = placeholderGo.AddComponent<TextMeshProUGUI>();
            placeholderText.text = "메시지를 입력하세요 (Enter로 전송)";
            placeholderText.color = new Color(0.5f, 0.5f, 0.5f, 0.8f);
            placeholderText.fontSize = 16;
            placeholderText.fontStyle = FontStyles.Italic;
            placeholderText.alignment = TextAlignmentOptions.MidlineLeft;

            // text component
            var textCompGo = CreateUIObject("Text", textArea.transform);
            SetRectFull(textCompGo, 0f, 0f, 1f, 1f, 0, 0, 0, 0);
            var textComp = textCompGo.AddComponent<TextMeshProUGUI>();
            textComp.text = "";
            textComp.color = TextPrimary;
            textComp.fontSize = 16;
            textComp.alignment = TextAlignmentOptions.MidlineLeft;

            var inputField = inputGo.AddComponent<TMP_InputField>();
            inputField.textViewport = textArea.GetComponent<RectTransform>();
            inputField.textComponent = textComp;
            inputField.placeholder = placeholderText;
            inputField.lineType = TMP_InputField.LineType.SingleLine;
            inputField.caretColor = TextPrimary;
            inputField.selectionColor = new Color(0.4f, 0.55f, 0.85f, 0.5f);

            // Send Button (right)
            var btnGo = CreateUIObject("SendButton", bar.transform);
            SetRectFull(btnGo, 0.80f, 0f, 1f, 1f, 0, 0, 0, 0);
            var btnImg = btnGo.AddComponent<Image>();
            btnImg.color = new Color(0.30f, 0.45f, 0.75f, 1f);
            var btn = btnGo.AddComponent<Button>();
            var colors = btn.colors;
            colors.normalColor = new Color(0.30f, 0.45f, 0.75f, 1f);
            colors.highlightedColor = new Color(0.40f, 0.55f, 0.85f, 1f);
            colors.pressedColor = new Color(0.20f, 0.35f, 0.65f, 1f);
            colors.selectedColor = colors.highlightedColor;
            btn.colors = colors;
            btn.targetGraphic = btnImg;

            var btnLabel = CreateUIObject("Label", btnGo.transform);
            SetRectFull(btnLabel, 0f, 0f, 1f, 1f, 0, 0, 0, 0);
            var btnText = btnLabel.AddComponent<TextMeshProUGUI>();
            btnText.text = "전송";
            btnText.color = Color.white;
            btnText.fontSize = 16;
            btnText.fontStyle = FontStyles.Bold;
            btnText.alignment = TextAlignmentOptions.Center;

            return (inputField, btn);
        }

        // ============ Interaction Prompt ============
        static GameObject BuildInteractionPrompt(Canvas canvas)
        {
            // root + Manager
            var manager = CreateUIObject("InteractionPromptManager", canvas.transform);
            var managerRt = manager.GetComponent<RectTransform>();
            managerRt.anchorMin = Vector2.zero;
            managerRt.anchorMax = Vector2.one;
            managerRt.offsetMin = Vector2.zero;
            managerRt.offsetMax = Vector2.zero;

            // PromptRoot — 화면 중앙 약간 하단
            var promptRoot = CreateUIObject("PromptRoot", manager.transform);
            var rootRt = promptRoot.GetComponent<RectTransform>();
            rootRt.anchorMin = new Vector2(0.30f, 0.42f);
            rootRt.anchorMax = new Vector2(0.70f, 0.50f);
            rootRt.offsetMin = Vector2.zero;
            rootRt.offsetMax = Vector2.zero;

            // 배경 — 검은색 반투명, outline 없음 (깔끔)
            var bg = promptRoot.AddComponent<Image>();
            bg.color = new Color(0f, 0f, 0f, 0.65f);

            var cg = promptRoot.AddComponent<CanvasGroup>();
            cg.alpha = 0f;
            cg.blocksRaycasts = false;
            cg.interactable = false;

            // 양 옆 짧은 액센트 라인 (좌)
            var leftLine = CreateUIObject("LeftAccent", promptRoot.transform);
            var leftRt = leftLine.GetComponent<RectTransform>();
            leftRt.anchorMin = new Vector2(0.02f, 0.48f);
            leftRt.anchorMax = new Vector2(0.12f, 0.52f);
            leftRt.offsetMin = Vector2.zero;
            leftRt.offsetMax = Vector2.zero;
            var leftImg = leftLine.AddComponent<Image>();
            leftImg.color = AccentNpcName;

            // 양 옆 짧은 액센트 라인 (우)
            var rightLine = CreateUIObject("RightAccent", promptRoot.transform);
            var rightRt = rightLine.GetComponent<RectTransform>();
            rightRt.anchorMin = new Vector2(0.88f, 0.48f);
            rightRt.anchorMax = new Vector2(0.98f, 0.52f);
            rightRt.offsetMin = Vector2.zero;
            rightRt.offsetMax = Vector2.zero;
            var rightImg = rightLine.AddComponent<Image>();
            rightImg.color = AccentNpcName;

            // Prompt text — 그림자만 (outline 없음)
            var textGo = CreateUIObject("PromptText", promptRoot.transform);
            SetRectFull(textGo, 0.14f, 0f, 0.86f, 1f, 0, 0, 4, 4);
            var text = textGo.AddComponent<TextMeshProUGUI>();
            text.text = "???과 얘기하기 [F]";
            text.color = TextPrimary;
            text.fontSize = 24;
            text.fontStyle = FontStyles.Bold;
            text.alignment = TextAlignmentOptions.Center;
            var shadow = textGo.AddComponent<Shadow>();
            shadow.effectColor = new Color(0f, 0f, 0f, 0.85f);
            shadow.effectDistance = new Vector2(2, -2);

            // Manager 컴포넌트 부착 + 슬롯 연결
            var mgr = manager.AddComponent<InteractionPromptManager>();
            mgr.promptRoot = promptRoot;
            mgr.promptText = text;
            mgr.canvasGroup = cg;

            // 시작 시 안 보이게
            promptRoot.SetActive(false);

            Undo.RegisterCreatedObjectUndo(manager, "Create InteractionPrompt");
            return manager;
        }

        // ============ Town Name Display ============
        static GameObject BuildTownNameDisplay(Canvas canvas)
        {
            var go = CreateUIObject("TownNameDisplay", canvas.transform);
            var rt = go.GetComponent<RectTransform>();
            // 화면 최상단에 붙게
            rt.anchorMin = new Vector2(0.20f, 0.78f);
            rt.anchorMax = new Vector2(0.80f, 0.98f);
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;

            var cg = go.AddComponent<CanvasGroup>();
            cg.alpha = 0f;
            cg.blocksRaycasts = false;
            cg.interactable = false;

            // 장식 — 상단 라인
            var topLine = CreateUIObject("TopLine", go.transform);
            var topRt = topLine.GetComponent<RectTransform>();
            topRt.anchorMin = new Vector2(0.15f, 0.78f);
            topRt.anchorMax = new Vector2(0.85f, 0.80f);
            topRt.offsetMin = Vector2.zero;
            topRt.offsetMax = Vector2.zero;
            var topImg = topLine.AddComponent<Image>();
            topImg.color = AccentNpcName;

            // Title (큰 글씨)
            var titleGo = CreateUIObject("TitleText", go.transform);
            SetRectFull(titleGo, 0f, 0.32f, 1f, 0.78f, 0, 0, 0, 0);
            var titleText = titleGo.AddComponent<TextMeshProUGUI>();
            titleText.text = "린덴브룩";
            titleText.color = TextPrimary;
            titleText.fontSize = 84;
            titleText.fontStyle = FontStyles.Bold;
            titleText.alignment = TextAlignmentOptions.Center;
            titleText.enableWordWrapping = false;

            // 그림자 효과 (Shadow 컴포넌트)
            var titleShadow = titleGo.AddComponent<Shadow>();
            titleShadow.effectColor = new Color(0f, 0f, 0f, 0.7f);
            titleShadow.effectDistance = new Vector2(2, -2);

            // Subtitle (작은 글씨, 영문 등)
            var subGo = CreateUIObject("SubtitleText", go.transform);
            SetRectFull(subGo, 0f, 0.10f, 1f, 0.32f, 0, 0, 0, 0);
            var subText = subGo.AddComponent<TextMeshProUGUI>();
            subText.text = "Lindenbrück";
            subText.color = AccentNpcName;
            subText.fontSize = 26;
            subText.fontStyle = FontStyles.Italic;
            subText.alignment = TextAlignmentOptions.Center;
            subText.enableWordWrapping = false;

            // 장식 — 하단 라인
            var botLine = CreateUIObject("BottomLine", go.transform);
            var botRt = botLine.GetComponent<RectTransform>();
            botRt.anchorMin = new Vector2(0.30f, 0.05f);
            botRt.anchorMax = new Vector2(0.70f, 0.07f);
            botRt.offsetMin = Vector2.zero;
            botRt.offsetMax = Vector2.zero;
            var botImg = botLine.AddComponent<Image>();
            botImg.color = new Color(AccentNpcName.r, AccentNpcName.g, AccentNpcName.b, 0.7f);

            // 컴포넌트 부착 + 슬롯 연결
            var disp = go.AddComponent<TownNameDisplay>();
            disp.titleText = titleText;
            disp.subtitleText = subText;
            disp.canvasGroup = cg;
            disp.townName = "린덴브룩";
            disp.subtitle = "Lindenbrück";
            disp.deactivateAfterPlay = false;  // T 키로 재생 가능하게 GameObject 유지
            disp.replayKey = KeyCode.T;

            Undo.RegisterCreatedObjectUndo(go, "Create TownNameDisplay");
            return go;
        }

        // ============ 헬퍼 ============
        /// <summary>
        /// Anchor min/max로 RectTransform 비례 위치 + offset(픽셀 단위 내부 패딩).
        /// offsetL,R: 좌/우 offset (양수 = 안쪽으로 들임)
        /// offsetT,B: 상/하 offset (양수 = 안쪽으로 들임)
        /// </summary>
        static void SetRectFull(GameObject go, float xMin, float yMin, float xMax, float yMax,
                                float offsetL, float offsetR, float offsetT, float offsetB)
        {
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(xMin, yMin);
            rt.anchorMax = new Vector2(xMax, yMax);
            rt.offsetMin = new Vector2(offsetL, offsetB);
            rt.offsetMax = new Vector2(-offsetR, -offsetT);
        }

        static GameObject CreateUIObject(string name, Transform parent)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent, false);
            return go;
        }
    }
}
#endif
