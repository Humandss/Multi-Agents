using System.Collections;
using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace NpcChat
{
    /// <summary>
    /// 자율 대화 발생 시 화면 우상단에 토스트 알림. fade in/out.
    /// GameTimeController.OnTickCompleted 구독.
    /// </summary>
    public class NpcEventToast : MonoBehaviour
    {
        [Header("표시")]
        public CanvasGroup canvasGroup;
        public TMP_Text titleLabel;
        public TMP_Text bodyLabel;

        [Header("타이밍")]
        public float fadeInDuration = 0.3f;
        public float holdDuration = 8f;
        public float fadeOutDuration = 0.8f;

        [Header("내용")]
        public int maxLines = 4;
        public int charLimit = 60;

        Coroutine _routine;

        void Start()
        {
            if (canvasGroup != null) canvasGroup.alpha = 0f;
            if (GameTimeController.Instance != null)
                GameTimeController.Instance.OnTickCompleted += HandleTick;
        }

        void OnDestroy()
        {
            if (GameTimeController.Instance != null)
                GameTimeController.Instance.OnTickCompleted -= HandleTick;
        }

        void HandleTick(ServerMessage msg)
        {
            if (msg.turns == null || msg.turns.Length == 0) return;
            if (string.IsNullOrEmpty(msg.npc_a) || string.IsNullOrEmpty(msg.npc_b)) return;

            // 화제 + 처음 N턴 표시
            if (titleLabel != null)
                titleLabel.text = $"Day {msg.day} · {msg.npc_a} ↔ {msg.npc_b}";

            if (bodyLabel != null)
            {
                var sb = new StringBuilder();
                int n = Mathf.Min(msg.turns.Length, maxLines);
                for (int i = 0; i < n; i++)
                {
                    var t = msg.turns[i];
                    string name = string.IsNullOrEmpty(t.speaker_ko) ? t.speaker : t.speaker_ko;
                    string body = t.text.Length > charLimit
                        ? t.text.Substring(0, charLimit) + "…"
                        : t.text;
                    sb.Append($"<b>{name}</b>: {body}\n");
                }
                bodyLabel.text = sb.ToString().TrimEnd('\n');
            }

            if (_routine != null) StopCoroutine(_routine);
            _routine = StartCoroutine(FadeRoutine());
        }

        IEnumerator FadeRoutine()
        {
            if (canvasGroup == null) yield break;

            // fade in
            float t = 0f;
            while (t < fadeInDuration)
            {
                t += Time.unscaledDeltaTime;
                canvasGroup.alpha = Mathf.Clamp01(t / fadeInDuration);
                yield return null;
            }
            canvasGroup.alpha = 1f;

            yield return new WaitForSecondsRealtime(holdDuration);

            // fade out
            t = 0f;
            while (t < fadeOutDuration)
            {
                t += Time.unscaledDeltaTime;
                canvasGroup.alpha = 1f - Mathf.Clamp01(t / fadeOutDuration);
                yield return null;
            }
            canvasGroup.alpha = 0f;
            _routine = null;
        }
    }
}
