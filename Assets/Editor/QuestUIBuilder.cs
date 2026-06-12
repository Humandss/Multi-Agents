#if UNITY_EDITOR
using NpcChat;
using TMPro;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

namespace NpcChat.EditorTools
{
    /// <summary>
    /// 좌측 퀘스트 리스트 UI 자동 생성.
    /// 메뉴: Tools > NpcChat > Create Quest List UI
    ///
    /// 생성:
    ///   QuestManager (싱글톤 GO, 없으면)
    ///   Canvas
    ///     └─ QuestListUI (루트, 항상 active — 이벤트 구독 유지)
    ///         └─ Panel (좌측 0.02~0.32, 대화 시작 시 표시)
    ///             ├─ Bg + 좌측 accent
    ///             ├─ Header "퀘스트 일지"
    ///             ├─ Scroll (Viewport+RectMask2D → Content+VerticalLayout)
    ///             └─ EmptyLabel
    /// </summary>
    public static class QuestUIBuilder
    {
        static readonly Color BgDark   = new Color(0.06f, 0.07f, 0.09f, 0.92f);
        static readonly Color Accent   = new Color(0.95f, 0.78f, 0.35f, 1f);
        static readonly Color Header   = new Color(1f, 0.9f, 0.55f, 1f);
        static readonly Color TextDim  = new Color(0.6f, 0.62f, 0.65f, 1f);

        [MenuItem("Tools/NpcChat/Create Quest List UI", priority = 11)]
        public static void Create()
        {
            DialogueUIBuilder.EnsureEventSystem();
            var canvas = DialogueUIBuilder.FindOrCreateCanvas();

            // QuestManager 싱글톤 GO
            if (Object.FindObjectOfType<QuestManager>() == null)
            {
                var mgr = new GameObject("QuestManager", typeof(QuestManager));
                Undo.RegisterCreatedObjectUndo(mgr, "Create QuestManager");
            }

            // 기존 패널 재생성
            var existing = canvas.transform.Find("QuestListUI");
            if (existing != null) Object.DestroyImmediate(existing.gameObject);

            // ── 루트 (항상 active, 비주얼 없음)
            var root = new GameObject("QuestListUI", typeof(RectTransform));
            root.transform.SetParent(canvas.transform, false);
            var rootRt = root.GetComponent<RectTransform>();
            rootRt.anchorMin = Vector2.zero;
            rootRt.anchorMax = Vector2.one;
            rootRt.offsetMin = Vector2.zero;
            rootRt.offsetMax = Vector2.zero;

            // ── Panel (좌측)
            var panel = new GameObject("Panel", typeof(RectTransform), typeof(Image));
            panel.transform.SetParent(root.transform, false);
            var panelRt = panel.GetComponent<RectTransform>();
            panelRt.anchorMin = new Vector2(0.02f, 0.05f);
            panelRt.anchorMax = new Vector2(0.32f, 0.78f);
            panelRt.offsetMin = Vector2.zero;
            panelRt.offsetMax = Vector2.zero;
            panel.GetComponent<Image>().color = BgDark;

            // 좌측 accent 라인
            var accent = new GameObject("Accent", typeof(RectTransform), typeof(Image));
            accent.transform.SetParent(panel.transform, false);
            var aRt = accent.GetComponent<RectTransform>();
            aRt.anchorMin = new Vector2(0, 0);
            aRt.anchorMax = new Vector2(0, 1);
            aRt.pivot = new Vector2(0, 0.5f);
            aRt.anchoredPosition = Vector2.zero;
            aRt.sizeDelta = new Vector2(4, 0);
            accent.GetComponent<Image>().color = Accent;

            // Header
            var header = new GameObject("Header", typeof(RectTransform));
            header.transform.SetParent(panel.transform, false);
            var hRt = header.GetComponent<RectTransform>();
            hRt.anchorMin = new Vector2(0, 1);
            hRt.anchorMax = new Vector2(1, 1);
            hRt.pivot = new Vector2(0.5f, 1);
            hRt.anchoredPosition = new Vector2(0, -8);
            hRt.sizeDelta = new Vector2(-24, 34);
            var headerLabel = header.AddComponent<TextMeshProUGUI>();
            headerLabel.text = "퀘스트 일지";
            headerLabel.fontSize = 20;
            headerLabel.color = Header;
            headerLabel.fontStyle = FontStyles.Bold;

            // Scroll 영역
            var scrollGo = new GameObject("Scroll", typeof(RectTransform), typeof(ScrollRect));
            scrollGo.transform.SetParent(panel.transform, false);
            var sRt = scrollGo.GetComponent<RectTransform>();
            sRt.anchorMin = new Vector2(0, 0);
            sRt.anchorMax = new Vector2(1, 1);
            sRt.offsetMin = new Vector2(10, 10);
            sRt.offsetMax = new Vector2(-10, -46);
            var scroll = scrollGo.GetComponent<ScrollRect>();
            scroll.horizontal = false;
            scroll.vertical = true;
            scroll.movementType = ScrollRect.MovementType.Clamped;
            scroll.scrollSensitivity = 25f;

            // Viewport (RectMask2D — Mask 가림 이슈 회피)
            var viewport = new GameObject("Viewport", typeof(RectTransform), typeof(RectMask2D));
            viewport.transform.SetParent(scrollGo.transform, false);
            var vRt = viewport.GetComponent<RectTransform>();
            vRt.anchorMin = Vector2.zero;
            vRt.anchorMax = Vector2.one;
            vRt.offsetMin = Vector2.zero;
            vRt.offsetMax = Vector2.zero;
            scroll.viewport = vRt;

            // Content (VerticalLayoutGroup + ContentSizeFitter)
            var content = new GameObject("Content", typeof(RectTransform));
            content.transform.SetParent(viewport.transform, false);
            var cRt = content.GetComponent<RectTransform>();
            cRt.anchorMin = new Vector2(0, 1);
            cRt.anchorMax = new Vector2(1, 1);
            cRt.pivot = new Vector2(0.5f, 1);
            cRt.anchoredPosition = Vector2.zero;
            cRt.sizeDelta = new Vector2(0, 0);
            scroll.content = cRt;

            var layout = content.AddComponent<VerticalLayoutGroup>();
            layout.spacing = 8;
            layout.childForceExpandHeight = false;
            layout.childForceExpandWidth = true;
            layout.childControlHeight = true;
            layout.childControlWidth = true;
            var fitter = content.AddComponent<ContentSizeFitter>();
            fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;

            // Empty label
            var empty = new GameObject("EmptyLabel", typeof(RectTransform));
            empty.transform.SetParent(panel.transform, false);
            var eRt = empty.GetComponent<RectTransform>();
            eRt.anchorMin = new Vector2(0, 0.5f);
            eRt.anchorMax = new Vector2(1, 0.5f);
            eRt.pivot = new Vector2(0.5f, 0.5f);
            eRt.anchoredPosition = Vector2.zero;
            eRt.sizeDelta = new Vector2(-30, 60);
            var emptyLabel = empty.AddComponent<TextMeshProUGUI>();
            emptyLabel.text = "받은 퀘스트가 없습니다.\nNPC와 친해지면 부탁을 받을 수 있어요.";
            emptyLabel.fontSize = 13;
            emptyLabel.color = TextDim;
            emptyLabel.alignment = TextAlignmentOptions.Center;

            // ── 시작 확인 모달 ("「제목」 시작하시겠습니까? 예/아니오")
            var modal = new GameObject("ConfirmModal", typeof(RectTransform), typeof(Image));
            modal.transform.SetParent(root.transform, false);
            var mRt = modal.GetComponent<RectTransform>();
            mRt.anchorMin = Vector2.zero;
            mRt.anchorMax = Vector2.one;
            mRt.offsetMin = Vector2.zero;
            mRt.offsetMax = Vector2.zero;
            // 뒤 클릭 차단용 반투명 풀스크린
            var modalBg = modal.GetComponent<Image>();
            modalBg.color = new Color(0, 0, 0, 0.55f);
            modalBg.raycastTarget = true;

            var box = new GameObject("Box", typeof(RectTransform), typeof(Image));
            box.transform.SetParent(modal.transform, false);
            var boxRt = box.GetComponent<RectTransform>();
            boxRt.anchorMin = new Vector2(0.5f, 0.5f);
            boxRt.anchorMax = new Vector2(0.5f, 0.5f);
            boxRt.pivot = new Vector2(0.5f, 0.5f);
            boxRt.anchoredPosition = Vector2.zero;
            boxRt.sizeDelta = new Vector2(380, 150);
            box.GetComponent<Image>().color = new Color(0.10f, 0.12f, 0.16f, 0.98f);

            var modalTitleGo = new GameObject("Title", typeof(RectTransform));
            modalTitleGo.transform.SetParent(box.transform, false);
            var mtRt = modalTitleGo.GetComponent<RectTransform>();
            mtRt.anchorMin = new Vector2(0, 1);
            mtRt.anchorMax = new Vector2(1, 1);
            mtRt.pivot = new Vector2(0.5f, 1);
            mtRt.anchoredPosition = new Vector2(0, -14);
            mtRt.sizeDelta = new Vector2(-28, 70);
            var modalTitleLabel = modalTitleGo.AddComponent<TextMeshProUGUI>();
            modalTitleLabel.text = "「퀘스트」\n시작하시겠습니까?";
            modalTitleLabel.fontSize = 17;
            modalTitleLabel.color = new Color(0.95f, 0.95f, 0.95f);
            modalTitleLabel.alignment = TextAlignmentOptions.Center;

            Button MakeModalBtn(string name, string label, Color bg, float anchorX)
            {
                var bgo = new GameObject(name, typeof(RectTransform), typeof(Image), typeof(Button));
                bgo.transform.SetParent(box.transform, false);
                bgo.GetComponent<Image>().color = bg;
                var brt = bgo.GetComponent<RectTransform>();
                brt.anchorMin = new Vector2(anchorX, 0);
                brt.anchorMax = new Vector2(anchorX, 0);
                brt.pivot = new Vector2(0.5f, 0);
                brt.anchoredPosition = new Vector2(0, 14);
                brt.sizeDelta = new Vector2(120, 34);
                var lgo = new GameObject("Label", typeof(RectTransform));
                lgo.transform.SetParent(bgo.transform, false);
                var lrt = lgo.GetComponent<RectTransform>();
                lrt.anchorMin = Vector2.zero;
                lrt.anchorMax = Vector2.one;
                lrt.offsetMin = Vector2.zero;
                lrt.offsetMax = Vector2.zero;
                var ltxt = lgo.AddComponent<TextMeshProUGUI>();
                ltxt.text = label;
                ltxt.fontSize = 15;
                ltxt.fontStyle = FontStyles.Bold;
                ltxt.color = Color.white;
                ltxt.alignment = TextAlignmentOptions.Center;
                return bgo.GetComponent<Button>();
            }

            var yesBtn = MakeModalBtn("YesBtn", "예", new Color(0.2f, 0.5f, 0.85f), 0.28f);
            var noBtn = MakeModalBtn("NoBtn", "아니오", new Color(0.35f, 0.36f, 0.40f), 0.72f);
            modal.SetActive(false);   // 평소 숨김 — QuestListUI.OpenConfirm이 켬

            // ── 컴포넌트 부착 + 슬롯 연결
            var ui = root.AddComponent<QuestListUI>();
            ui.panelRoot = panel;
            ui.contentRoot = content.transform;
            ui.emptyLabel = emptyLabel;
            ui.modalRoot = modal;
            ui.modalTitle = modalTitleLabel;
            ui.modalYesBtn = yesBtn;
            ui.modalNoBtn = noBtn;

            // 한국어 폰트 (정적 텍스트 + 런타임 생성용 슬롯 둘 다)
            var koreanFont = DialogueUIBuilder.FindKoreanFontAsset();
            if (koreanFont != null)
            {
                DialogueUIBuilder.ApplyFontToAllTMP(root, koreanFont);
                ui.koreanFont = koreanFont;   // 런타임 생성 엔트리용
                Debug.Log($"[QuestUIBuilder] 한국어 폰트 적용: {koreanFont.name}");
            }
            else
            {
                Debug.LogWarning("[QuestUIBuilder] 한국어 TMP 폰트 못 찾음 — 런타임 엔트리 글자 깨질 수 있음");
            }

            EditorSceneManager.MarkSceneDirty(root.scene);
            Selection.activeGameObject = root;
            Debug.Log("[QuestUIBuilder] 퀘스트 리스트 UI 생성 완료. 대화 시작 시 좌측에 표시됩니다.");
        }
    }
}
#endif
