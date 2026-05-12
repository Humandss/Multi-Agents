using System.Collections;
using TMPro;
using UnityEngine;

namespace NpcChat
{
    /// <summary>
    /// 화면 중앙 상호작용 안내 UI (싱글톤).
    /// NpcInteractor가 플레이어 근처 들어오면 Show("???과 얘기하기 [F]") 호출.
    /// 멀어지면 Hide() 호출. 대화 중에는 자동 숨김.
    /// </summary>
    public class InteractionPromptManager : MonoBehaviour
    {
        public static InteractionPromptManager Instance { get; private set; }

        [Header("UI Refs")]
        [Tooltip("Prompt UI 루트 (보임/숨김 토글)")]
        public GameObject promptRoot;
        [Tooltip("안내 텍스트")]
        public TMP_Text promptText;
        [Tooltip("페이드용 (없어도 됨)")]
        public CanvasGroup canvasGroup;

        [Header("Settings")]
        [Tooltip("페이드 시간 (초)")]
        public float fadeDuration = 0.15f;
        [Tooltip("대화 중이면 자동 숨김")]
        public bool hideDuringDialogue = true;

        private Coroutine _fadeRoutine;
        private string _currentText = "";
        private bool _shown = false;

        void Awake()
        {
            if (Instance != null && Instance != this) { Destroy(gameObject); return; }
            Instance = this;
            if (canvasGroup != null) canvasGroup.alpha = 0f;
            if (promptRoot != null) promptRoot.SetActive(false);
        }

        void Update()
        {
            // 대화 중이면 강제 숨김
            if (hideDuringDialogue && _shown
                && DialogueManager.Instance != null
                && DialogueManager.Instance.IsInDialogue)
            {
                Hide();
            }
        }

        public void Show(string text)
        {
            if (_shown && _currentText == text) return;
            _currentText = text;
            _shown = true;
            if (promptText != null) promptText.text = text;
            if (promptRoot != null) promptRoot.SetActive(true);
            StartFade(1f, null);
        }

        public void Hide()
        {
            if (!_shown) return;
            _shown = false;
            _currentText = "";
            StartFade(0f, () =>
            {
                if (promptRoot != null) promptRoot.SetActive(false);
            });
        }

        private void StartFade(float target, System.Action onComplete)
        {
            if (canvasGroup == null)
            {
                onComplete?.Invoke();
                return;
            }
            if (_fadeRoutine != null) StopCoroutine(_fadeRoutine);
            _fadeRoutine = StartCoroutine(FadeCoroutine(target, onComplete));
        }

        private IEnumerator FadeCoroutine(float target, System.Action onComplete)
        {
            float start = canvasGroup.alpha;
            if (fadeDuration <= 0f)
            {
                canvasGroup.alpha = target;
                onComplete?.Invoke();
                yield break;
            }
            float t = 0f;
            while (t < fadeDuration)
            {
                t += Time.deltaTime;
                canvasGroup.alpha = Mathf.Lerp(start, target, t / fadeDuration);
                yield return null;
            }
            canvasGroup.alpha = target;
            onComplete?.Invoke();
        }
    }
}
