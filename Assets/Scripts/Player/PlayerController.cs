using UnityEngine;

namespace NpcChat
{
    /// <summary>
    /// 3D 플레이어 이동 + 점프 + 마우스 회전
    /// </summary>
    [RequireComponent(typeof(Rigidbody))]
    public class PlayerController : MonoBehaviour
    {
        [Header("Movement")]
        [Tooltip("이동 속도")]
        public float moveSpeed = 5f;
        [Tooltip("점프 힘")]
        public float jumpForce = 7f;

        [Header("Mouse Look")]
        [Tooltip("마우스 감도")]
        public float mouseSensitivity = 2f;
        [Tooltip("카메라 상하 각도 최소")]
        public float pitchMin = -70f;
        [Tooltip("카메라 상하 각도 최대")]
        public float pitchMax = 70f;
        [Tooltip("게임 시작 시 커서 lock")]
        public bool lockCursorOnStart = true;
        [Tooltip("Player 자식 카메라 (상하 회전 적용)")]
        public Transform cameraTransform;

        [Header("Jump")]
        [Tooltip("발 아래 충돌면 인식 임계값 (normal.y > 이 값 → ground로 인정). 0.5 = 약 60도 경사까지 ground.")]
        [Range(0f, 1f)]
        public float groundNormalThreshold = 0.5f;

        [Header("Input Lock")]
        [Tooltip("대화/Quest UI 열려있을 때 외부에서 true → 이동/회전/점프 차단")]
        public bool inputLocked = false;

        private Rigidbody _rb;
        private bool _isGrounded;
        private Vector3 _moveDirection;
        private float _pitch = 0f;  // 카메라 누적 X 회전
        private bool _canJump = true;  // 공중에선 false, 착지하면 true

        // Spawn 위치/회전 — Start 시점에 저장. 시간 진행 시 복귀.
        private Vector3 _spawnPos;
        private Quaternion _spawnRot;

        void Awake()
        {
            _rb = GetComponent<Rigidbody>();
            // 모든 회전 freeze — 마우스로만 직접 회전 (충돌/물리 영향 차단)
            _rb.constraints = RigidbodyConstraints.FreezeRotation;

            // 한글 IME 모드 비활성 — 알파벳 키(WASD, F)가 자모로 변환되지 않도록.
            // (Windows 한글 사용자가 한/영 토글 안 해도 게임 키 정상 작동)
            Input.imeCompositionMode = IMECompositionMode.Off;
        }

        void Start()
        {
            if (lockCursorOnStart)
                LockCursor(true);

            // Spawn 위치 저장 — 시간 진행(tick) 시 복귀용
            _spawnPos = transform.position;
            _spawnRot = transform.rotation;

            // 시간 진행 시작 시 spawn으로 복귀 + 입력 잠금. 완료/실패 시 잠금 해제.
            if (GameTimeController.Instance != null)
            {
                GameTimeController.Instance.OnTickStarted += HandleTickStarted;
                GameTimeController.Instance.OnTickCompleted += HandleTickCompleted;
                GameTimeController.Instance.OnTickFailed += HandleTickFailed;
            }
        }

        void OnDestroy()
        {
            if (GameTimeController.Instance != null)
            {
                GameTimeController.Instance.OnTickStarted -= HandleTickStarted;
                GameTimeController.Instance.OnTickCompleted -= HandleTickCompleted;
                GameTimeController.Instance.OnTickFailed -= HandleTickFailed;
            }
        }

        void HandleTickStarted()
        {
            ReturnToSpawn();
            inputLocked = true;
        }

        void HandleTickCompleted(ServerMessage msg) { inputLocked = false; }
        void HandleTickFailed(string err) { inputLocked = false; }

        public void ReturnToSpawn()
        {
            transform.position = _spawnPos;
            transform.rotation = _spawnRot;
            if (_rb != null)
            {
                _rb.velocity = Vector3.zero;
                _rb.angularVelocity = Vector3.zero;
            }
            _pitch = 0f;
            if (cameraTransform != null)
                cameraTransform.localRotation = Quaternion.identity;
            _moveDirection = Vector3.zero;
        }

        void Update()
        {
            // ESC는 DialogueManager가 전용 처리. PlayerController는 ESC 무시.
            // 디버그용 cursor unlock이 필요하면 F12 등 다른 키 사용.
            if (Input.GetKeyDown(KeyCode.F12))
                LockCursor(Cursor.lockState != CursorLockMode.Locked);

            if (inputLocked)
            {
                _moveDirection = Vector3.zero;
                return;
            }

            // Mouse Look (커서 lock 상태일 때만)
            if (Cursor.lockState == CursorLockMode.Locked)
            {
                float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity;
                float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity;

                // 플레이어 Y축 회전 (좌우) — Rigidbody.MoveRotation으로 물리 충돌
                Quaternion deltaYaw = Quaternion.AngleAxis(mouseX, Vector3.up);
                _rb.MoveRotation(_rb.rotation * deltaYaw);

                // 카메라 X축 회전 (상하 pitch) — Player 자식 Camera에 localRotation
                if (cameraTransform != null)
                {
                    _pitch -= mouseY;
                    _pitch = Mathf.Clamp(_pitch, pitchMin, pitchMax);
                    cameraTransform.localRotation = Quaternion.Euler(_pitch, 0f, 0f);
                }
            }

            // 입력 — Input.GetKey 직접 (InputField focus 영향 안 받음)
            float h = 0f, v = 0f;
            if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow))    v += 1f;
            if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow))  v -= 1f;
            if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow))  h -= 1f;
            if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow)) h += 1f;
            Vector3 input = (transform.forward * v + transform.right * h);
            if (input.sqrMagnitude > 1f) input.Normalize();
            _moveDirection = input;

            // 점프 — 착지 상태일 때만 가능
            if (_canJump && Input.GetKeyDown(KeyCode.Space))
            {
                Vector3 vel = _rb.velocity;
                vel.y = jumpForce;
                _rb.velocity = vel;
                _canJump = false;  // 다시 착지할 때까지 점프 X
            }
        }

        void FixedUpdate()
        {
            if (inputLocked) return;
            // 수평 이동만 적용 (Y velocity 보존 — 점프/낙하)
            Vector3 v = _rb.velocity;
            v.x = _moveDirection.x * moveSpeed;
            v.z = _moveDirection.z * moveSpeed;
            _rb.velocity = v;
        }

        // 충돌체와 닿아있는 동안 매 frame 호출 — 발 아래 충돌면이 있으면 _canJump = true
        void OnCollisionStay(Collision collision)
        {
            foreach (var contact in collision.contacts)
            {
                // contact.normal.y > 임계값 → 위쪽 면 = ground
                if (contact.normal.y > groundNormalThreshold)
                {
                    _canJump = true;
                    _isGrounded = true;
                    return;
                }
            }
        }

        void OnCollisionExit(Collision collision)
        {
            _isGrounded = false;
        }

        private void LockCursor(bool locked)
        {
            Cursor.lockState = locked ? CursorLockMode.Locked : CursorLockMode.None;
            Cursor.visible = !locked;
        }
    }
}
