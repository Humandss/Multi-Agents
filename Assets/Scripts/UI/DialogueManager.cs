using System.Collections;
using UnityEngine;

namespace NpcChat
{
    /// <summary>
    /// 대화 시작/종료 + 카메라 전환 + UI 관리 (싱글톤, Scene에 1개).
    /// </summary>
    public class DialogueManager : MonoBehaviour
    {
        public static DialogueManager Instance { get; private set; }

        [Header("References")]
        [Tooltip("플레이어 — inputLocked 토글용")]
        public PlayerController player;
        [Tooltip("Player 자식 카메라 Transform — localPos/localRot 조작")]
        public Transform cameraTransform;
        [Tooltip("대화창 UI 패널 — 평소 비활성, 대화 시 활성")]
        public GameObject dialoguePanel;

        [Header("Camera Transition")]
        [Tooltip("NPC와 카메라 거리 (정면 기준)")]
        public float cameraDistance = 2.5f;
        [Tooltip("NPC 머리 높이 offset (LookAt 타겟)")]
        public float lookHeightOffset = 1.2f;
        [Tooltip("NPC를 화면 왼쪽으로 밀기 위한 카메라 우측 offset (양수 = 화면 왼쪽으로 NPC)")]
        public float cameraRightOffset = 0.7f;
        [Tooltip("카메라 전환 시간 (초)")]
        public float transitionDuration = 0.8f;

        // 내부 상태
        private NpcInteractor _currentNpc;
        private Coroutine _transitionRoutine;
        private Vector3 _originalCamLocalPos;
        private Quaternion _originalCamLocalRot;
        private bool _inDialogue = false;

        public bool IsInDialogue => _inDialogue;
        public NpcInteractor CurrentNpc => _currentNpc;

        // 이벤트 — DialogueController 등이 구독
        public event System.Action<NpcInteractor> OnDialogueStarted;
        public event System.Action OnDialogueEnded;

        void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
        }

        void Start()
        {
            // Inspector 잘못 연결 가드
            if (cameraTransform == null && Camera.main != null)
            {
                cameraTransform = Camera.main.transform;
                Debug.Log("[DialogueManager] cameraTransform 자동 설정 = Camera.main");
            }
            if (player != null && cameraTransform != null && cameraTransform == player.transform)
            {
                // 자동 fallback: Player 자식 중 Camera 컴포넌트 찾기
                var cam = player.GetComponentInChildren<Camera>();
                if (cam != null && cam.transform != player.transform)
                {
                    cameraTransform = cam.transform;
                    Debug.Log($"[DialogueManager] cameraTransform 자동 교정 → {cam.name}");
                }
            }

            if (cameraTransform != null)
            {
                _originalCamLocalPos = cameraTransform.localPosition;
                _originalCamLocalRot = cameraTransform.localRotation;
            }
            if (dialoguePanel != null) dialoguePanel.SetActive(false);
        }

        void Update()
        {
            // ESC: 대화 종료
            if (_inDialogue && Input.GetKeyDown(KeyCode.Escape))
                EndDialogue();
        }

        public void StartDialogue(NpcInteractor npc)
        {
            if (_inDialogue) return;
            _currentNpc = npc;
            _inDialogue = true;

            // Player 입력 차단 (마우스 회전, 이동, 점프 다 막힘)
            if (player != null) player.inputLocked = true;

            // 대화 중 커서 보이게 (UI 클릭 가능하게)
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;

            // NPC wander 정지 + Player 쪽 바라봄
            var wanderer = npc.GetComponent<NpcWanderer>();
            if (wanderer != null && player != null)
            {
                wanderer.PauseWandering();
                wanderer.LookAt(player.transform.position);
            }

            // UI 활성
            if (dialoguePanel != null) dialoguePanel.SetActive(true);

            // 카메라 전환
            if (cameraTransform != null && player != null)
            {
                Vector3 targetLocalPos;
                Quaternion targetLocalRot;
                ComputeDialogueCameraLocal(npc.transform, out targetLocalPos, out targetLocalRot);

                if (_transitionRoutine != null) StopCoroutine(_transitionRoutine);
                _transitionRoutine = StartCoroutine(LerpCamera(targetLocalPos, targetLocalRot, null));
            }

            Debug.Log($"[Dialogue] {npc.npcName} 대화 시작");
            OnDialogueStarted?.Invoke(npc);
        }

        public void EndDialogue()
        {
            if (!_inDialogue) return;
            _inDialogue = false;

            // UI 비활성
            if (dialoguePanel != null) dialoguePanel.SetActive(false);

            // 커서 다시 lock (게임으로 복귀)
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;

            // NPC wander 재개
            if (_currentNpc != null)
            {
                var wanderer = _currentNpc.GetComponent<NpcWanderer>();
                if (wanderer != null) wanderer.ResumeWandering();
            }

            // input 즉시 복구 (lerp 완료까지 기다리지 않음 — 마우스 회전 바로 가능)
            // 카메라 lerp는 별도로 진행. lerp 중엔 카메라 localRotation을 매 frame 덮어쓰지만
            // lerp 끝나면 _originalCamLocalRot으로 돌아오고 그 후 PlayerController의 pitch 적용됨.
            if (player != null) player.inputLocked = false;

            // 카메라 원래 자리로 복귀
            if (cameraTransform != null)
            {
                if (_transitionRoutine != null) StopCoroutine(_transitionRoutine);
                _transitionRoutine = StartCoroutine(LerpCamera(
                    _originalCamLocalPos, _originalCamLocalRot, onComplete: null
                ));
            }

            Debug.Log("[Dialogue] 종료");
            OnDialogueEnded?.Invoke();
            _currentNpc = null;
        }

        /// <summary>
        /// NPC 정면 + 약간 측면으로 카메라 local pos/rot 계산.
        /// NPC를 화면 왼쪽에 두기 위해 카메라가 살짝 오른쪽으로 panning.
        /// </summary>
        private void ComputeDialogueCameraLocal(Transform npcTransform, out Vector3 localPos, out Quaternion localRot)
        {
            Vector3 npcCenter = npcTransform.position + Vector3.up * lookHeightOffset;
            Vector3 playerPos = player.transform.position;
            Vector3 dirFromNpcToPlayer = (playerPos - npcTransform.position);
            dirFromNpcToPlayer.y = 0;
            if (dirFromNpcToPlayer.sqrMagnitude < 0.0001f)
                dirFromNpcToPlayer = -npcTransform.forward;
            dirFromNpcToPlayer.Normalize();

            // World target — NPC 정면(player 쪽) cameraDistance 거리
            Vector3 targetWorldPos = npcCenter + dirFromNpcToPlayer * cameraDistance;

            // NPC를 화면 왼쪽으로 두기 위해 카메라를 우측 panning
            Vector3 cameraRight = Vector3.Cross(Vector3.up, -dirFromNpcToPlayer).normalized;
            targetWorldPos += cameraRight * cameraRightOffset;

            Quaternion targetWorldRot = Quaternion.LookRotation(npcCenter - targetWorldPos);

            // Player 자식 좌표계로 변환
            Transform parent = cameraTransform.parent;
            if (parent != null)
            {
                localPos = parent.InverseTransformPoint(targetWorldPos);
                localRot = Quaternion.Inverse(parent.rotation) * targetWorldRot;
            }
            else
            {
                localPos = targetWorldPos;
                localRot = targetWorldRot;
            }
        }

        IEnumerator LerpCamera(Vector3 targetLocalPos, Quaternion targetLocalRot, System.Action onComplete)
        {
            Vector3 startPos = cameraTransform.localPosition;
            Quaternion startRot = cameraTransform.localRotation;
            float t = 0f;
            while (t < transitionDuration)
            {
                t += Time.deltaTime;
                float a = Mathf.SmoothStep(0f, 1f, t / transitionDuration);
                cameraTransform.localPosition = Vector3.Lerp(startPos, targetLocalPos, a);
                cameraTransform.localRotation = Quaternion.Slerp(startRot, targetLocalRot, a);
                yield return null;
            }
            cameraTransform.localPosition = targetLocalPos;
            cameraTransform.localRotation = targetLocalRot;
            onComplete?.Invoke();
        }
    }
}
