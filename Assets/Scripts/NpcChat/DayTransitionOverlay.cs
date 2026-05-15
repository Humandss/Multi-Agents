using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace NpcChat
{
    /// <summary>
    /// 시간 진행 (tick) 동안 화면 전체 검은 페이드 + 회전 시계 + dots 애니메이션.
    /// GameTimeController 이벤트 구독.
    /// </summary>
    public class DayTransitionOverlay : MonoBehaviour
    {
        [Header("UI 슬롯")]
        public CanvasGroup canvasGroup;
        public RectTransform clockHand;     // 회전할 시계 바늘 (Image)
        public TMP_Text titleLabel;          // "마을 시간 흐르는 중..." / "Day N"
        public TMP_Text dotsLabel;           // "..." 애니메이션
        public TMP_Text subtitleLabel;       // 부가 정보 (전파 N건, NPC ↔ NPC)

        [Header("타이밍")]
        public float fadeInDuration = 0.5f;
        public float fadeOutDuration = 0.8f;
        public float dayDisplayHold = 1.2f;  // tick 완료 후 Day N 보여주는 시간
        public float dotIntervalSec = 0.35f;

        [Header("시계")]
        public float clockRotationSpeed = 540f;  // deg/sec (시계 방향)

        bool _active;
        Coroutine _fadeRoutine;

        void Awake()
        {
            if (canvasGroup != null)
            {
                canvasGroup.alpha = 0f;
                canvasGroup.blocksRaycasts = false;
                canvasGroup.interactable = false;
            }
        }

        void Start()
        {
            if (GameTimeController.Instance != null)
            {
                GameTimeController.Instance.OnTickStarted += HandleStart;
                GameTimeController.Instance.OnTickCompleted += HandleComplete;
                GameTimeController.Instance.OnTickFailed += HandleFail;
            }
        }

        void OnDestroy()
        {
            if (GameTimeController.Instance != null)
            {
                GameTimeController.Instance.OnTickStarted -= HandleStart;
                GameTimeController.Instance.OnTickCompleted -= HandleComplete;
                GameTimeController.Instance.OnTickFailed -= HandleFail;
            }
        }

        void Update()
        {
            if (!_active) return;
            // 시계 바늘 회전 (Time.unscaledDeltaTime — 게임 멈춤 영향 없음)
            if (clockHand != null)
                clockHand.Rotate(Vector3.forward, -clockRotationSpeed * Time.unscaledDeltaTime);
            // dots 애니메이션: . → .. → ... → "" 반복
            if (dotsLabel != null)
            {
                int phase = (int)(Time.unscaledTime / dotIntervalSec) % 4;
                dotsLabel.text = new string('.', phase);
            }
        }

        void HandleStart()
        {
            _active = true;
            if (titleLabel != null) titleLabel.text = "마을 시간이 흐르는 중";
            if (subtitleLabel != null) subtitleLabel.text = "NPC들이 소식을 주고 받고 있어요";
            if (canvasGroup != null) canvasGroup.blocksRaycasts = true;

            // Quest 카드가 열려있으면 닫음 (검은 화면 위로 quest 보이지 않게)
            CloseQuestCard();

            if (_fadeRoutine != null) StopCoroutine(_fadeRoutine);
            _fadeRoutine = StartCoroutine(Fade(0f, 1f, fadeInDuration));
        }

        /// <summary>
        /// Scene에서 quest 카드 찾아 비활성화. NpcChatDemoUI가 inactive여도 작동.
        /// </summary>
        void CloseQuestCard()
        {
            // 1) 활성 NpcChatDemoUI에서 questCard 참조 시도
            var ui = FindObjectOfType<NpcChatDemoUI>(true);
            if (ui != null && ui.questCard != null)
            {
                ui.questCard.SetActive(false);
                return;
            }
            // 2) Fallback: 이름으로 찾기
            var card = GameObject.Find("QuestCard");
            if (card != null) card.SetActive(false);
        }

        void HandleComplete(ServerMessage msg)
        {
            if (titleLabel != null) titleLabel.text = $"Day {msg.day}";
            // Subtitle: 전파 건수 + NPC 대화 페어 (있을 시)
            if (subtitleLabel != null)
            {
                int evCount = msg.events != null ? msg.events.Length : 0;
                string sub = $"전파 {evCount}건";
                if (msg.turns != null && msg.turns.Length > 0
                    && !string.IsNullOrEmpty(msg.npc_a) && !string.IsNullOrEmpty(msg.npc_b))
                {
                    sub += $"  ·  {msg.npc_a} ↔ {msg.npc_b}";
                }
                subtitleLabel.text = sub;
            }
            if (_fadeRoutine != null) StopCoroutine(_fadeRoutine);
            _fadeRoutine = StartCoroutine(HoldThenFadeOut());
        }

        void HandleFail(string err)
        {
            if (titleLabel != null) titleLabel.text = "시간 진행 실패";
            if (subtitleLabel != null) subtitleLabel.text = err.Length > 60 ? err.Substring(0, 60) + "..." : err;
            if (_fadeRoutine != null) StopCoroutine(_fadeRoutine);
            _fadeRoutine = StartCoroutine(HoldThenFadeOut());
        }

        IEnumerator HoldThenFadeOut()
        {
            yield return new WaitForSecondsRealtime(dayDisplayHold);
            yield return Fade(canvasGroup != null ? canvasGroup.alpha : 1f, 0f, fadeOutDuration);
            _active = false;
            if (canvasGroup != null) canvasGroup.blocksRaycasts = false;
            if (dotsLabel != null) dotsLabel.text = "";
        }

        IEnumerator Fade(float from, float to, float duration)
        {
            if (canvasGroup == null) yield break;
            float t = 0f;
            while (t < duration)
            {
                t += Time.unscaledDeltaTime;
                canvasGroup.alpha = Mathf.Lerp(from, to, t / duration);
                yield return null;
            }
            canvasGroup.alpha = to;
        }
    }
}
