#if UNITY_EDITOR
using NpcChat;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace NpcChat.EditorTools
{
    /// <summary>
    /// Day HUD + 자율 대화 토스트 알림 자동 빌드.
    /// 메뉴: Tools > NpcChat > Create Game Time HUD + Toast
    ///
    /// 생성:
    ///   Canvas
    ///     ├─ GameTimeHUD (좌상단 "Day N" + [N] 시간 진행 + status)
    ///     ├─ NpcEventToast (우상단 fade-in 알림)
    ///     └─ GameTimeController (singleton)
    /// </summary>
    public static class GameTimeUIBuilder
    {
        static readonly Color BgDark    = new Color(0.05f, 0.06f, 0.08f, 0.78f);
        static readonly Color BgAccent  = new Color(0.10f, 0.12f, 0.16f, 0.92f);
        static readonly Color TextDay   = new Color(1f, 0.92f, 0.65f, 1f);
        static readonly Color TextHint  = new Color(0.70f, 0.72f, 0.75f, 1f);
        static readonly Color TextStatus = new Color(0.78f, 0.85f, 0.78f, 1f);
        static readonly Color ToastTitle = new Color(1f, 0.88f, 0.42f, 1f);
        static readonly Color ToastBody  = new Color(0.92f, 0.92f, 0.92f, 1f);
        static readonly Color ToastBorder = new Color(0.45f, 0.50f, 0.65f, 0.6f);

        [MenuItem("Tools/NpcChat/Create Game Time HUD + Toast", priority = 10)]
        public static void CreateAll()
        {
            DialogueUIBuilder.EnsureEventSystem();
            var canvas = DialogueUIBuilder.FindOrCreateCanvas();

            var controller = EnsureController();
            var hud = BuildHud(canvas, controller);
            var toast = BuildToast(canvas, controller);
            var overlay = BuildDayTransitionOverlay(canvas);

            var koreanFont = DialogueUIBuilder.FindKoreanFontAsset();
            if (koreanFont != null)
            {
                DialogueUIBuilder.ApplyFontToAllTMP(hud, koreanFont);
                DialogueUIBuilder.ApplyFontToAllTMP(toast, koreanFont);
                DialogueUIBuilder.ApplyFontToAllTMP(overlay, koreanFont);
                Debug.Log($"[GameTimeUIBuilder] 한국어 폰트 적용: {koreanFont.name}");
            }

            EditorSceneManager.MarkSceneDirty(hud.scene);
            Selection.activeGameObject = hud;
            Debug.Log("[GameTimeUIBuilder] HUD + Toast + DayTransitionOverlay 생성 완료. [N]으로 시간 진행.");
        }

        // ============ GameTimeController singleton ============
        static GameTimeController EnsureController()
        {
            var existing = Object.FindObjectOfType<GameTimeController>();
            if (existing != null) return existing;

            var go = new GameObject("GameTimeController",
                typeof(GameTimeController));
            Undo.RegisterCreatedObjectUndo(go, "Create GameTimeController");
            return go.GetComponent<GameTimeController>();
        }

        // ============ 좌상단 Day HUD ============
        static GameObject BuildHud(Canvas canvas, GameTimeController controller)
        {
            var existing = canvas.transform.Find("GameTimeHUD");
            if (existing != null)
            {
                Object.DestroyImmediate(existing.gameObject);
            }

            var root = new GameObject("GameTimeHUD",
                typeof(RectTransform), typeof(CanvasGroup));
            root.transform.SetParent(canvas.transform, false);
            var rt = root.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(0, 1);
            rt.anchorMax = new Vector2(0, 1);
            rt.pivot = new Vector2(0, 1);
            rt.anchoredPosition = new Vector2(24, -24);
            rt.sizeDelta = new Vector2(280, 88);

            // 배경 패널
            var bg = new GameObject("Bg", typeof(RectTransform), typeof(Image));
            bg.transform.SetParent(root.transform, false);
            var bgRt = bg.GetComponent<RectTransform>();
            bgRt.anchorMin = Vector2.zero; bgRt.anchorMax = Vector2.one;
            bgRt.offsetMin = Vector2.zero; bgRt.offsetMax = Vector2.zero;
            bg.GetComponent<Image>().color = BgDark;

            // 가로 accent 라인 (좌측)
            var accent = new GameObject("Accent", typeof(RectTransform), typeof(Image));
            accent.transform.SetParent(root.transform, false);
            var aRt = accent.GetComponent<RectTransform>();
            aRt.anchorMin = new Vector2(0, 0); aRt.anchorMax = new Vector2(0, 1);
            aRt.pivot = new Vector2(0, 0.5f);
            aRt.anchoredPosition = new Vector2(0, 0);
            aRt.sizeDelta = new Vector2(4, 0);
            accent.GetComponent<Image>().color = new Color(0.95f, 0.78f, 0.35f, 1f);

            // Day 텍스트
            var day = new GameObject("Day", typeof(RectTransform));
            day.transform.SetParent(root.transform, false);
            var dRt = day.GetComponent<RectTransform>();
            dRt.anchorMin = new Vector2(0, 1); dRt.anchorMax = new Vector2(1, 1);
            dRt.pivot = new Vector2(0, 1);
            dRt.anchoredPosition = new Vector2(16, -10);
            dRt.sizeDelta = new Vector2(-16, 32);
            var dayLabel = day.AddComponent<TextMeshProUGUI>();
            dayLabel.text = "Day 0";
            dayLabel.fontSize = 24;
            dayLabel.color = TextDay;
            dayLabel.fontStyle = FontStyles.Bold;

            // hint 텍스트
            var hint = new GameObject("Hint", typeof(RectTransform));
            hint.transform.SetParent(root.transform, false);
            var hRt = hint.GetComponent<RectTransform>();
            hRt.anchorMin = new Vector2(0, 1); hRt.anchorMax = new Vector2(1, 1);
            hRt.pivot = new Vector2(0, 1);
            hRt.anchoredPosition = new Vector2(16, -42);
            hRt.sizeDelta = new Vector2(-16, 20);
            var hintLabel = hint.AddComponent<TextMeshProUGUI>();
            hintLabel.text = "[N] 시간 진행";
            hintLabel.fontSize = 14;
            hintLabel.color = TextHint;

            // status 텍스트
            var status = new GameObject("Status", typeof(RectTransform));
            status.transform.SetParent(root.transform, false);
            var sRt = status.GetComponent<RectTransform>();
            sRt.anchorMin = new Vector2(0, 1); sRt.anchorMax = new Vector2(1, 1);
            sRt.pivot = new Vector2(0, 1);
            sRt.anchoredPosition = new Vector2(16, -62);
            sRt.sizeDelta = new Vector2(-16, 22);
            var statusLabel = status.AddComponent<TextMeshProUGUI>();
            statusLabel.text = "";
            statusLabel.fontSize = 13;
            statusLabel.color = TextStatus;
            statusLabel.richText = true;

            // HUD 컴포넌트 부착 + 슬롯 연결
            var hud = root.AddComponent<GameTimeHud>();
            hud.dayLabel = dayLabel;
            hud.hintLabel = hintLabel;
            hud.statusLabel = statusLabel;

            Undo.RegisterCreatedObjectUndo(root, "Create GameTimeHud");
            return root;
        }

        // ============ 우상단 자율 대화 토스트 ============
        static GameObject BuildToast(Canvas canvas, GameTimeController controller)
        {
            var existing = canvas.transform.Find("NpcEventToast");
            if (existing != null)
            {
                Object.DestroyImmediate(existing.gameObject);
            }

            var root = new GameObject("NpcEventToast",
                typeof(RectTransform), typeof(CanvasGroup));
            root.transform.SetParent(canvas.transform, false);
            var rt = root.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(1, 1);
            rt.anchorMax = new Vector2(1, 1);
            rt.pivot = new Vector2(1, 1);
            rt.anchoredPosition = new Vector2(-24, -24);
            rt.sizeDelta = new Vector2(440, 200);

            var cg = root.GetComponent<CanvasGroup>();
            cg.alpha = 0f;
            cg.interactable = false;
            cg.blocksRaycasts = false;

            // 배경
            var bg = new GameObject("Bg", typeof(RectTransform), typeof(Image));
            bg.transform.SetParent(root.transform, false);
            var bgRt = bg.GetComponent<RectTransform>();
            bgRt.anchorMin = Vector2.zero; bgRt.anchorMax = Vector2.one;
            bgRt.offsetMin = Vector2.zero; bgRt.offsetMax = Vector2.zero;
            bg.GetComponent<Image>().color = BgAccent;

            // border accent (좌측 세로 라인)
            var border = new GameObject("Border", typeof(RectTransform), typeof(Image));
            border.transform.SetParent(root.transform, false);
            var bdRt = border.GetComponent<RectTransform>();
            bdRt.anchorMin = new Vector2(0, 0); bdRt.anchorMax = new Vector2(0, 1);
            bdRt.pivot = new Vector2(0, 0.5f);
            bdRt.anchoredPosition = Vector2.zero;
            bdRt.sizeDelta = new Vector2(4, 0);
            border.GetComponent<Image>().color = ToastBorder;

            // title
            var title = new GameObject("Title", typeof(RectTransform));
            title.transform.SetParent(root.transform, false);
            var tRt = title.GetComponent<RectTransform>();
            tRt.anchorMin = new Vector2(0, 1); tRt.anchorMax = new Vector2(1, 1);
            tRt.pivot = new Vector2(0, 1);
            tRt.anchoredPosition = new Vector2(16, -12);
            tRt.sizeDelta = new Vector2(-32, 28);
            var titleLabel = title.AddComponent<TextMeshProUGUI>();
            titleLabel.text = "Day 0 · NPC ↔ NPC";
            titleLabel.fontSize = 18;
            titleLabel.color = ToastTitle;
            titleLabel.fontStyle = FontStyles.Bold;

            // body
            var body = new GameObject("Body", typeof(RectTransform));
            body.transform.SetParent(root.transform, false);
            var bdyRt = body.GetComponent<RectTransform>();
            bdyRt.anchorMin = Vector2.zero; bdyRt.anchorMax = Vector2.one;
            bdyRt.pivot = new Vector2(0.5f, 0.5f);
            bdyRt.offsetMin = new Vector2(16, 12); bdyRt.offsetMax = new Vector2(-16, -44);
            var bodyLabel = body.AddComponent<TextMeshProUGUI>();
            bodyLabel.text = "";
            bodyLabel.fontSize = 14;
            bodyLabel.color = ToastBody;
            bodyLabel.richText = true;
            bodyLabel.alignment = TextAlignmentOptions.TopLeft;
            bodyLabel.enableWordWrapping = true;

            var toast = root.AddComponent<NpcEventToast>();
            toast.canvasGroup = cg;
            toast.titleLabel = titleLabel;
            toast.bodyLabel = bodyLabel;

            Undo.RegisterCreatedObjectUndo(root, "Create NpcEventToast");
            return root;
        }

        // ============ Day Transition Overlay (검은 페이드 + 시계 + dots) ============
        static GameObject BuildDayTransitionOverlay(Canvas canvas)
        {
            var existing = canvas.transform.Find("DayTransitionOverlay");
            if (existing != null)
            {
                Object.DestroyImmediate(existing.gameObject);
            }

            // Root — 풀스크린 검은 영역
            var root = new GameObject("DayTransitionOverlay",
                typeof(RectTransform), typeof(CanvasGroup));
            root.transform.SetParent(canvas.transform, false);
            var rt = root.GetComponent<RectTransform>();
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
            var cg = root.GetComponent<CanvasGroup>();
            cg.alpha = 0f;
            cg.blocksRaycasts = false;
            cg.interactable = false;

            // 검은 배경 — 거의 완전 검은 (alpha 0.98)
            var bg = new GameObject("BlackBg", typeof(RectTransform), typeof(Image));
            bg.transform.SetParent(root.transform, false);
            var bgRt = bg.GetComponent<RectTransform>();
            bgRt.anchorMin = Vector2.zero; bgRt.anchorMax = Vector2.one;
            bgRt.offsetMin = Vector2.zero; bgRt.offsetMax = Vector2.zero;
            bg.GetComponent<Image>().color = new Color(0, 0, 0, 0.98f);

            // 시계 컨테이너 (가운데)
            var clockBox = new GameObject("ClockBox", typeof(RectTransform));
            clockBox.transform.SetParent(root.transform, false);
            var cbRt = clockBox.GetComponent<RectTransform>();
            cbRt.anchorMin = new Vector2(0.5f, 0.5f);
            cbRt.anchorMax = new Vector2(0.5f, 0.5f);
            cbRt.pivot = new Vector2(0.5f, 0.5f);
            cbRt.anchoredPosition = new Vector2(0, 40);
            cbRt.sizeDelta = new Vector2(120, 120);

            // 시계 외곽 원 (Image with circle sprite would be ideal, but Unity 기본은 사각.
            //  대신 outline 효과로 원형 느낌)
            var clockFace = new GameObject("ClockFace", typeof(RectTransform), typeof(Image));
            clockFace.transform.SetParent(clockBox.transform, false);
            var cfRt = clockFace.GetComponent<RectTransform>();
            cfRt.anchorMin = Vector2.zero; cfRt.anchorMax = Vector2.one;
            cfRt.offsetMin = Vector2.zero; cfRt.offsetMax = Vector2.zero;
            var faceImg = clockFace.GetComponent<Image>();
            faceImg.color = new Color(0.95f, 0.85f, 0.55f, 0.2f);
            // Unity 기본 UI Sprite 중 원형: "Knob" (built-in)
            var knobSprite = Resources.GetBuiltinResource<Sprite>("UI/Skin/Knob.psd");
            if (knobSprite != null) faceImg.sprite = knobSprite;

            // 시계 바늘 (회전할 막대)
            var hand = new GameObject("ClockHand", typeof(RectTransform), typeof(Image));
            hand.transform.SetParent(clockBox.transform, false);
            var handRt = hand.GetComponent<RectTransform>();
            handRt.anchorMin = new Vector2(0.5f, 0.5f);
            handRt.anchorMax = new Vector2(0.5f, 0.5f);
            handRt.pivot = new Vector2(0.5f, 0f);   // 막대 아래쪽이 회전 중심
            handRt.anchoredPosition = Vector2.zero;
            handRt.sizeDelta = new Vector2(4, 44);  // 가는 막대
            hand.GetComponent<Image>().color = new Color(1f, 0.92f, 0.55f, 1f);

            // 중심 점 (시계 한가운데)
            var center = new GameObject("Center", typeof(RectTransform), typeof(Image));
            center.transform.SetParent(clockBox.transform, false);
            var centerRt = center.GetComponent<RectTransform>();
            centerRt.anchorMin = new Vector2(0.5f, 0.5f);
            centerRt.anchorMax = new Vector2(0.5f, 0.5f);
            centerRt.pivot = new Vector2(0.5f, 0.5f);
            centerRt.anchoredPosition = Vector2.zero;
            centerRt.sizeDelta = new Vector2(10, 10);
            var centerImg = center.GetComponent<Image>();
            centerImg.color = new Color(1f, 0.92f, 0.55f, 1f);
            if (knobSprite != null) centerImg.sprite = knobSprite;

            // Title — "마을 시간이 흐르는 중" / "Day N"
            var title = new GameObject("Title", typeof(RectTransform));
            title.transform.SetParent(root.transform, false);
            var titleRt = title.GetComponent<RectTransform>();
            titleRt.anchorMin = new Vector2(0.5f, 0.5f);
            titleRt.anchorMax = new Vector2(0.5f, 0.5f);
            titleRt.pivot = new Vector2(0.5f, 0.5f);
            titleRt.anchoredPosition = new Vector2(0, -60);
            titleRt.sizeDelta = new Vector2(600, 50);
            var titleLabel = title.AddComponent<TextMeshProUGUI>();
            titleLabel.text = "마을 시간이 흐르는 중";
            titleLabel.fontSize = 28;
            titleLabel.color = new Color(1f, 0.92f, 0.65f, 1f);
            titleLabel.fontStyle = FontStyles.Bold;
            titleLabel.alignment = TextAlignmentOptions.Center;

            // Dots — "..." 애니메이션
            var dots = new GameObject("Dots", typeof(RectTransform));
            dots.transform.SetParent(root.transform, false);
            var dotsRt = dots.GetComponent<RectTransform>();
            dotsRt.anchorMin = new Vector2(0.5f, 0.5f);
            dotsRt.anchorMax = new Vector2(0.5f, 0.5f);
            dotsRt.pivot = new Vector2(0.5f, 0.5f);
            dotsRt.anchoredPosition = new Vector2(0, -110);
            dotsRt.sizeDelta = new Vector2(400, 50);
            var dotsLabel = dots.AddComponent<TextMeshProUGUI>();
            dotsLabel.text = "...";
            dotsLabel.fontSize = 40;
            dotsLabel.color = new Color(0.95f, 0.78f, 0.35f, 1f);
            dotsLabel.fontStyle = FontStyles.Bold;
            dotsLabel.alignment = TextAlignmentOptions.Center;

            // Subtitle — 전파 N건 · NPC ↔ NPC
            var sub = new GameObject("Subtitle", typeof(RectTransform));
            sub.transform.SetParent(root.transform, false);
            var subRt = sub.GetComponent<RectTransform>();
            subRt.anchorMin = new Vector2(0.5f, 0.5f);
            subRt.anchorMax = new Vector2(0.5f, 0.5f);
            subRt.pivot = new Vector2(0.5f, 0.5f);
            subRt.anchoredPosition = new Vector2(0, -160);
            subRt.sizeDelta = new Vector2(700, 30);
            var subLabel = sub.AddComponent<TextMeshProUGUI>();
            subLabel.text = "NPC들이 소식을 주고 받고 있어요";
            subLabel.fontSize = 16;
            subLabel.color = new Color(0.78f, 0.78f, 0.78f, 1f);
            subLabel.alignment = TextAlignmentOptions.Center;

            // DayTransitionOverlay 컴포넌트 부착 + 슬롯 연결
            var overlay = root.AddComponent<DayTransitionOverlay>();
            overlay.canvasGroup = cg;
            overlay.clockHand = handRt;
            overlay.titleLabel = titleLabel;
            overlay.dotsLabel = dotsLabel;
            overlay.subtitleLabel = subLabel;

            // 다른 UI 위에 표시되도록 마지막 sibling
            root.transform.SetAsLastSibling();

            Undo.RegisterCreatedObjectUndo(root, "Create DayTransitionOverlay");
            return root;
        }
    }
}
#endif
