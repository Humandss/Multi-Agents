using UnityEngine;

namespace NpcChat
{
    /// <summary>
    /// 게임 시작 시 UI 패널들을 비활성화.
    /// 대화/Quest는 NPC 상호작용 시 활성화 (DialogueManager 등에서 처리).
    ///
    /// 사용법:
    /// 1. Canvas 아래에 빈 GameObject 만들기
    /// 2. UIController 컴포넌트 붙임
    /// 3. Inspector에서 비활성화할 패널들 연결
    /// </summary>
    public class UIController : MonoBehaviour
    {
        [Header("초기 비활성 UI 패널")]
        [Tooltip("NPC 대화창 — 평소엔 비활성, NPC 상호작용 시 활성")]
        public GameObject dialoguePanel;

        [Tooltip("Quest 표시 패널 — Quest 받을 때 활성")]
        public GameObject questPanel;

        [Tooltip("메모리 로그 (디버그용, 선택)")]
        public GameObject memoryLogPanel;

        [Tooltip("Day 진행 버튼 패널 (선택)")]
        public GameObject dayProgressPanel;

        [Header("초기 활성 UI 패널 (선택)")]
        [Tooltip("게임 시작 시 보여야 하는 HUD 등")]
        public GameObject[] initiallyActive;

        void Awake()
        {
            // 비활성화
            SetPanel(dialoguePanel, false);
            SetPanel(questPanel, false);
            SetPanel(memoryLogPanel, false);
            SetPanel(dayProgressPanel, false);

            // 활성화
            if (initiallyActive != null)
            {
                foreach (var go in initiallyActive)
                    SetPanel(go, true);
            }
        }

        private void SetPanel(GameObject panel, bool active)
        {
            if (panel != null)
                panel.SetActive(active);
        }
    }
}
