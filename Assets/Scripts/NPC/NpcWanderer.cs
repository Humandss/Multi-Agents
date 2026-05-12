using System.Collections;
using UnityEngine;

namespace NpcChat
{
    /// <summary>
    /// NPC가 시작 위치(anchor) 주변에서 일정 반경 내 랜덤 wander.
    /// 대화 시작 시 정지 (PauseWandering), 종료 시 재개 (ResumeWandering).
    /// </summary>
    public class NpcWanderer : MonoBehaviour
    {
        [Header("Wander Range")]
        [Tooltip("anchor(시작 위치) 기준 wander 반경")]
        public float wanderRadius = 3f;

        [Header("Movement")]
        public float moveSpeed = 1f;
        public float rotationSpeed = 180f;
        [Tooltip("도착 인정 거리")]
        public float arriveDistance = 0.2f;

        [Header("Idle")]
        public float idleMin = 1.5f;
        public float idleMax = 4f;

        private Vector3 _anchor;
        private Vector3 _destination;
        private bool _paused = false;

        void Awake()
        {
            // Rigidbody가 있으면 회전 freeze + isKinematic 설정 — 캡슐 안 넘어지게.
            // transform.position 직접 이동하니 isKinematic이 자연스러움 (물리 시뮬과 충돌 X).
            var rb = GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.constraints = RigidbodyConstraints.FreezeRotation;
                rb.isKinematic = true;
            }
        }

        void Start()
        {
            _anchor = transform.position;
            StartCoroutine(WanderLoop());
        }

        public void PauseWandering() { _paused = true; }
        public void ResumeWandering() { _paused = false; }

        /// <summary>대화 시작 시 NPC가 플레이어 쪽 바라보도록.</summary>
        public void LookAt(Vector3 worldPos)
        {
            Vector3 dir = worldPos - transform.position;
            dir.y = 0;
            if (dir.sqrMagnitude > 0.01f)
                transform.rotation = Quaternion.LookRotation(dir);
        }

        IEnumerator WanderLoop()
        {
            while (true)
            {
                // 대화 중 정지
                while (_paused) yield return null;

                // 새 랜덤 destination
                Vector2 offset2D = Random.insideUnitCircle * wanderRadius;
                _destination = _anchor + new Vector3(offset2D.x, 0, offset2D.y);

                // 이동
                while (Vector3.Distance(transform.position, _destination) > arriveDistance)
                {
                    if (_paused) { yield return null; continue; }

                    Vector3 dir = _destination - transform.position;
                    dir.y = 0;
                    if (dir.sqrMagnitude < 0.0001f) break;
                    dir.Normalize();

                    // 회전
                    Quaternion targetRot = Quaternion.LookRotation(dir);
                    transform.rotation = Quaternion.RotateTowards(
                        transform.rotation, targetRot, rotationSpeed * Time.deltaTime
                    );

                    // 위치 이동 (transform 직접 — NPC는 Rigidbody 없음 가정)
                    transform.position += dir * moveSpeed * Time.deltaTime;
                    yield return null;
                }

                // idle
                yield return new WaitForSeconds(Random.Range(idleMin, idleMax));
            }
        }

        void OnDrawGizmosSelected()
        {
            Vector3 center = Application.isPlaying ? _anchor : transform.position;
            Gizmos.color = new Color(0.2f, 1f, 0.2f, 0.4f);
            Gizmos.DrawWireSphere(center, wanderRadius);
        }
    }
}
