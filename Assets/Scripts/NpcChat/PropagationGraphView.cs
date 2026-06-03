using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace NpcChat
{
    /// <summary>
    /// 화면 우상단에 5 NPC 노드 그래프. 플레이어 발화에서 비롯된 정보가
    /// 전파될 때(TickEvent.player_origin) from→to로 점이 흐르는 펄스를 보여준다.
    ///
    /// GameTimeController.OnTickCompleted 구독.
    /// 사용법: 빈 GameObject 하나에 이 컴포넌트만 추가하면 런타임에 UI 자동 생성.
    /// (Scene 수작업 불필요. 위치/크기는 인스펙터 필드로 조정.)
    /// </summary>
    public class PropagationGraphView : MonoBehaviour
    {
        [Header("배치")]
        public Vector2 panelSize = new Vector2(280, 210);
        public Vector2 panelMargin = new Vector2(20, 20);  // 우상단 기준 여백
        public float nodeRadius = 72f;

        [Header("펄스")]
        public float pulseDuration = 1.2f;
        public Color playerColor = new Color(0.91f, 0.30f, 0.24f);  // 빨강 (플레이어 정보)
        public Color nodeColor = new Color(0.20f, 0.60f, 0.86f);    // 파랑 (NPC)

        // app.py display()가 첫 글자를 대문자로 보내므로 여기도 대문자.
        static readonly string[] NPCS = { "Mathilda", "Hermann", "Finn", "Bernhardt", "Elias" };

        readonly Dictionary<string, RectTransform> _nodes = new Dictionary<string, RectTransform>();
        RectTransform _root;

        void Start()
        {
            BuildUI();
            if (GameTimeController.Instance != null)
                GameTimeController.Instance.OnTickCompleted += HandleTick;
        }

        void OnDestroy()
        {
            if (GameTimeController.Instance != null)
                GameTimeController.Instance.OnTickCompleted -= HandleTick;
        }

        void BuildUI()
        {
            var canvasGo = new GameObject("PropagationCanvas");
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 50;
            canvasGo.AddComponent<CanvasScaler>();
            canvasGo.AddComponent<GraphicRaycaster>();

            var panel = NewRect("Panel", canvasGo.transform);
            panel.anchorMin = panel.anchorMax = new Vector2(1, 1);
            panel.pivot = new Vector2(1, 1);
            panel.sizeDelta = panelSize;
            panel.anchoredPosition = new Vector2(-panelMargin.x, -panelMargin.y);
            var bg = panel.gameObject.AddComponent<Image>();
            bg.color = new Color(0f, 0f, 0f, 0.45f);
            _root = panel;

            var title = NewText("Title", panel, "마을 정보 전파", 14);
            title.alignment = TextAlignmentOptions.Center;
            title.rectTransform.anchorMin = new Vector2(0, 1);
            title.rectTransform.anchorMax = new Vector2(1, 1);
            title.rectTransform.pivot = new Vector2(0.5f, 1);
            title.rectTransform.sizeDelta = new Vector2(0, 22);
            title.rectTransform.anchoredPosition = new Vector2(0, -6);

            var center = new Vector2(0, -18);
            for (int i = 0; i < NPCS.Length; i++)
            {
                float ang = Mathf.PI / 2f - i * (2f * Mathf.PI / NPCS.Length);
                var pos = center + new Vector2(Mathf.Cos(ang), Mathf.Sin(ang)) * nodeRadius;
                _nodes[NPCS[i]] = MakeNode(NPCS[i], pos);
            }
        }

        RectTransform MakeNode(string npcName, Vector2 pos)
        {
            var node = NewRect(npcName, _root);
            node.sizeDelta = new Vector2(54, 54);
            node.anchoredPosition = pos;
            var img = node.gameObject.AddComponent<Image>();
            img.color = nodeColor;

            var label = NewText(npcName + "_label", node, npcName, 10);
            label.alignment = TextAlignmentOptions.Center;
            label.rectTransform.anchorMin = Vector2.zero;
            label.rectTransform.anchorMax = Vector2.one;
            label.rectTransform.offsetMin = Vector2.zero;
            label.rectTransform.offsetMax = Vector2.zero;
            return node;
        }

        void HandleTick(ServerMessage msg)
        {
            if (msg == null || msg.events == null) return;
            float delay = 0f;
            foreach (var ev in msg.events)
            {
                if (ev == null || !ev.player_origin) continue;
                if (!_nodes.ContainsKey(ev.from) || !_nodes.ContainsKey(ev.to)) continue;
                StartCoroutine(Pulse(_nodes[ev.from], _nodes[ev.to], delay));
                delay += 0.25f;
            }
        }

        IEnumerator Pulse(RectTransform from, RectTransform to, float delay)
        {
            if (delay > 0f) yield return new WaitForSecondsRealtime(delay);

            var dot = NewRect("dot", _root);
            dot.sizeDelta = new Vector2(16, 16);
            var img = dot.gameObject.AddComponent<Image>();
            img.color = playerColor;

            float t = 0f;
            while (t < pulseDuration)
            {
                t += Time.unscaledDeltaTime;
                float u = Mathf.Clamp01(t / pulseDuration);
                dot.anchoredPosition = Vector2.Lerp(from.anchoredPosition, to.anchoredPosition, u);
                yield return null;
            }

            var toImg = to.GetComponent<Image>();
            if (toImg != null) StartCoroutine(Flash(toImg));
            Destroy(dot.gameObject);
        }

        IEnumerator Flash(Image img)
        {
            var orig = img.color;
            float t = 0f;
            while (t < 0.6f)
            {
                t += Time.unscaledDeltaTime;
                img.color = Color.Lerp(playerColor, orig, t / 0.6f);
                yield return null;
            }
            img.color = orig;
        }

        // ---------- UI 헬퍼 ----------
        static RectTransform NewRect(string objName, Transform parent)
        {
            var go = new GameObject(objName, typeof(RectTransform));
            go.transform.SetParent(parent, false);
            return go.GetComponent<RectTransform>();
        }

        static TMP_Text NewText(string objName, Transform parent, string content, float size)
        {
            var rect = NewRect(objName, parent);
            var t = rect.gameObject.AddComponent<TextMeshProUGUI>();
            t.text = content;
            t.fontSize = size;
            t.color = Color.white;
            return t;
        }
    }
}
