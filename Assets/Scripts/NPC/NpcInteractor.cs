using UnityEngine;

namespace NpcChat
{
    /// <summary>
    /// 플레이어가 가까이 오면 hint 표시, F 누르면 DialogueManager에 대화 시작 알림.
    /// 각 NPC GameObject에 붙임.
    /// </summary>
    public class NpcInteractor : MonoBehaviour
    {
        [Header("NPC Identity")]
        [Tooltip("서버 등록된 NPC 이름 (elias / hermann / mathilda / finn / bernhardt)")]
        public string npcName;

        [Header("Interaction")]
        [Tooltip("상호작용 가능 거리")]
        public float interactionRange = 2.5f;
        [Tooltip("상호작용 키")]
        public KeyCode interactionKey = KeyCode.F;

        [Header("UI Hint (선택)")]
        [Tooltip("\"F: 대화\" 등 머리 위 hint GameObject. 플레이어 가까이 오면 활성")]
        public GameObject interactionHint;

        [Header("Center Screen Prompt")]
        [Tooltip("화면 중앙에 \"이름과 얘기하기\" 안내. InteractionPromptManager 사용.")]
        public bool useCenterPrompt = true;
        [Tooltip("프롬프트 형식.")]
        public string promptFormat = "{npc}과 얘기하기 [{key}]";

        private Transform _player;
        private bool _isPlayerNear = false;

        void Start()
        {
            var playerGO = GameObject.FindGameObjectWithTag("Player");
            if (playerGO != null) _player = playerGO.transform;
            else Debug.LogWarning($"[NpcInteractor:{npcName}] Player 태그 GameObject 못 찾음");

            if (interactionHint != null) interactionHint.SetActive(false);
        }

        void Update()
        {
            if (_player == null) return;

            float dist = Vector3.Distance(transform.position, _player.position);
            bool nowNear = dist <= interactionRange;

            if (nowNear != _isPlayerNear)
            {
                _isPlayerNear = nowNear;
                if (interactionHint != null) interactionHint.SetActive(_isPlayerNear);

                // 중앙 화면 프롬프트
                if (useCenterPrompt && InteractionPromptManager.Instance != null)
                {
                    if (_isPlayerNear)
                    {
                        string text = promptFormat
                            .Replace("???", gameObject.name)
                            .Replace("{npc}", gameObject.name)
                            .Replace("{key}", interactionKey.ToString());
                        InteractionPromptManager.Instance.Show(text);
                    }
                    else
                    {
                        InteractionPromptManager.Instance.Hide();
                    }
                }
            }

            if (_isPlayerNear && Input.GetKeyDown(interactionKey))
            {
                if (DialogueManager.Instance != null)
                    DialogueManager.Instance.StartDialogue(this);
                else
                    Debug.LogWarning("[NpcInteractor] DialogueManager 인스턴스 없음 — Scene에 두었는지 확인");
            }
        }

        public Transform PlayerTransform => _player;

        void OnDrawGizmosSelected()
        {
            Gizmos.color = new Color(1f, 1f, 0f, 0.3f);
            Gizmos.DrawWireSphere(transform.position, interactionRange);
        }
    }
}
