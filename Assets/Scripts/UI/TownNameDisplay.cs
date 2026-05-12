using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace NpcChat
{
    /// <summary>
    /// 게임 시작 시 마을 이름을 화면 상단에 fade in/out으로 표시.
    /// RPG 게임에서 새 지역 진입 시 흔한 연출.
    /// </summary>
    public class TownNameDisplay : MonoBehaviour
    {
        [Header("Texts")]
        public TMP_Text titleText;
        public TMP_Text subtitleText;

        [Header("Content")]
        [Tooltip("마을 이름 (한국어)")]
        public string townName = "린덴브룩";
        [Tooltip("부제 (영문/설명 등, 비워두면 표시 안 함)")]
        public string subtitle = "Lindenbrück";

        [Header("Timing")]
        [Tooltip("시작 후 표시까지 대기 (초)")]
        public float startDelay = 0.5f;
        [Tooltip("페이드 인 시간")]
        public float fadeInDuration = 1.0f;
        [Tooltip("유지 시간")]
        public float holdDuration = 2.5f;
        [Tooltip("페이드 아웃 시간")]
        public float fadeOutDuration = 1.2f;

        [Header("Refs")]
        public CanvasGroup canvasGroup;

        [Header("Settings")]
        [Tooltip("Start에서 자동 재생")]
        public bool playOnStart = true;
        [Tooltip("표시 후 GameObject 비활성. 단 키로 재생하려면 false 권장.")]
        public bool deactivateAfterPlay = false;

        [Header("Replay Key")]
        [Tooltip("이 키 누르면 다시 재생")]
        public KeyCode replayKey = KeyCode.T;
        [Tooltip("InputField에 입력 중이면 무시 (채팅 입력 도중 T 글자가 trigger되지 않게)")]
        public bool ignoreWhenInputFocused = true;
        [Tooltip("대화 중이면 무시 (DialogueManager.IsInDialogue 체크)")]
        public bool ignoreWhenInDialogue = true;

        void Awake()
        {
            if (canvasGroup == null) canvasGroup = GetComponent<CanvasGroup>();
            if (canvasGroup != null) canvasGroup.alpha = 0;
            if (titleText != null) titleText.text = townName;
            if (subtitleText != null)
            {
                subtitleText.text = subtitle;
                subtitleText.gameObject.SetActive(!string.IsNullOrEmpty(subtitle));
            }
        }

        void Start()
        {
            // 디버그: 어떤 상태인지 출력
            Debug.Log($"[TownNameDisplay] Start. gameObject.activeInHierarchy={gameObject.activeInHierarchy}, " +
                      $"canvasGroup={(canvasGroup != null ? "OK" : "NULL")}, " +
                      $"titleText={(titleText != null ? "OK" : "NULL")}, " +
                      $"playOnStart={playOnStart}, deactivateAfterPlay={deactivateAfterPlay}");
            if (playOnStart) Play();
        }

        void OnDisable() { Debug.Log("[TownNameDisplay] OnDisable — GameObject 비활성화됨. T키 안 먹힘."); }
        void OnEnable()  { Debug.Log("[TownNameDisplay] OnEnable — GameObject 활성화됨."); }

        void Update()
        {
            if (!Input.GetKeyDown(replayKey)) return;

            Debug.Log($"[TownNameDisplay] {replayKey} 키 감지. " +
                      $"inputFocused={IsAnyInputFieldFocused()}, " +
                      $"inDialogue={(DialogueManager.Instance != null && DialogueManager.Instance.IsInDialogue)}");

            // 채팅 입력 중이면 무시 (InputField가 T 글자를 받아야 함)
            if (ignoreWhenInputFocused && IsAnyInputFieldFocused())
            {
                Debug.Log("[TownNameDisplay] InputField focused — skip");
                return;
            }

            // 대화 중이면 무시
            if (ignoreWhenInDialogue && DialogueManager.Instance != null
                && DialogueManager.Instance.IsInDialogue)
            {
                Debug.Log("[TownNameDisplay] In dialogue — skip");
                return;
            }

            Debug.Log("[TownNameDisplay] Replay!");
            Play();
        }

        static bool IsAnyInputFieldFocused()
        {
            var es = EventSystem.current;
            if (es == null) return false;
            var sel = es.currentSelectedGameObject;
            if (sel == null) return false;
            // TMP InputField 또는 legacy InputField
            if (sel.GetComponent<TMP_InputField>() != null) return true;
            if (sel.GetComponent<InputField>() != null) return true;
            return false;
        }

        /// <summary>외부에서 호출해서 다시 재생 가능 (예: 새 지역 진입).</summary>
        public void Play()
        {
            if (titleText != null) titleText.text = townName;
            if (subtitleText != null)
            {
                subtitleText.text = subtitle;
                subtitleText.gameObject.SetActive(!string.IsNullOrEmpty(subtitle));
            }
            gameObject.SetActive(true);
            StopAllCoroutines();
            StartCoroutine(ShowSequence());
        }

        IEnumerator ShowSequence()
        {
            if (canvasGroup == null)
            {
                Debug.LogWarning("[TownNameDisplay] CanvasGroup 없음");
                yield break;
            }

            canvasGroup.alpha = 0f;
            if (startDelay > 0f) yield return new WaitForSeconds(startDelay);

            // Fade in
            yield return Fade(0f, 1f, fadeInDuration);

            // Hold
            if (holdDuration > 0f) yield return new WaitForSeconds(holdDuration);

            // Fade out
            yield return Fade(1f, 0f, fadeOutDuration);

            // deactivateAfterPlay=true여도 T 키로 재생 가능하려면 GameObject 활성 유지해야 함.
            // 단 사용자가 명시적으로 true 설정한 경우엔 비활성화.
            if (deactivateAfterPlay)
            {
                Debug.LogWarning("[TownNameDisplay] Deactivate After Play=true라 비활성화. " +
                                 "T 키 재생 원하면 Inspector에서 체크 해제하세요.");
                gameObject.SetActive(false);
            }
        }

        IEnumerator Fade(float from, float to, float duration)
        {
            if (duration <= 0f)
            {
                canvasGroup.alpha = to;
                yield break;
            }
            float t = 0f;
            while (t < duration)
            {
                t += Time.deltaTime;
                canvasGroup.alpha = Mathf.SmoothStep(from, to, t / duration);
                yield return null;
            }
            canvasGroup.alpha = to;
        }
    }
}
