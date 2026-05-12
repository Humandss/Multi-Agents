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

            var koreanFont = DialogueUIBuilder.FindKoreanFontAsset();
            if (koreanFont != null)
            {
                DialogueUIBuilder.ApplyFontToAllTMP(hud, koreanFont);
                DialogueUIBuilder.ApplyFontToAllTMP(toast, koreanFont);
                Debug.Log($"[GameTimeUIBuilder] 한국어 폰트 적용: {koreanFont.name}");
            }

            EditorSceneManager.MarkSceneDirty(hud.scene);
            Selection.activeGameObject = hud;
            Debug.Log("[GameTimeUIBuilder] 게임 시간 HUD + Toast 생성 완료. [N]으로 시간 진행.");
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
    }
}
#endif
