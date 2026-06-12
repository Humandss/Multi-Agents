using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

namespace NpcChat
{
    public enum QuestState { Available, Accepted, Completed }

    /// <summary>퀘스트 1건 (서버 /quests/{npc} 기준).</summary>
    [Serializable]
    public class QuestEntry
    {
        public string id;
        public string title;
        public string description;
        public string reward;
        public string giver;          // npc 이름
        public int trustRequired;
        public bool eligible;         // 지금 시작 가능 (trust 충족 + available)
        public QuestState state = QuestState.Available;
    }

    /// <summary>
    /// NPC별 퀘스트 리스트 관리 (싱글톤). 서버가 source of truth:
    ///
    /// - FetchQuestsFor(npc): GET /quests/{npc} → 리스트 갱신 (대화 시작 시 호출)
    /// - CompleteQuest: POST /quest_complete → trust +10 (진행 중 quest만)
    /// - 수락/거절은 대화 흐름(WS quest_propose → 플레이어 답변)에서 처리되고,
    ///   결과(quest_stage)가 오면 NpcChatDemoUI가 다시 Fetch를 불러 동기화.
    /// </summary>
    public class QuestManager : MonoBehaviour
    {
        public static QuestManager Instance { get; private set; }

        [Header("서버")]
        public string serverHost = "127.0.0.1";
        public int serverPort = 8000;

        [Header("디버그/시연")]
        [Tooltip("진행 중인 모든 퀘스트 즉시 완료 (시연용 치트)")]
        public KeyCode completeAllKey = KeyCode.H;

        [Header("상태 (읽기용)")]
        public string currentNpc = "";
        public List<QuestEntry> quests = new List<QuestEntry>();

        public event Action OnQuestsChanged;
        /// <summary>완료 성공 시 (entry, NPC 반응 대사). 대화창 표시용.</summary>
        public event Action<QuestEntry, string> OnQuestCompleted;

        void Awake()
        {
            if (Instance != null && Instance != this) { Destroy(gameObject); return; }
            Instance = this;
        }

        void OnDestroy()
        {
            if (Instance == this) Instance = null;
        }

        void Update()
        {
            // H키 — 진행 중 퀘스트 전체 완료 (시연 치트). 채팅 입력 중엔 무시.
            if (Input.GetKeyDown(completeAllKey) && !IsInputFieldFocused())
                StartCoroutine(CompleteAllCoroutine());
        }

        static bool IsInputFieldFocused()
        {
            var sel = UnityEngine.EventSystems.EventSystem.current?.currentSelectedGameObject;
            if (sel == null) return false;
            if (sel.GetComponent<TMPro.TMP_InputField>() != null) return true;
            if (sel.GetComponent<UnityEngine.UI.InputField>() != null) return true;
            return false;
        }

        /// <summary>진행 중(수락한) 모든 퀘스트를 서버에서 완료 처리 (H키).</summary>
        IEnumerator CompleteAllCoroutine()
        {
            string url = $"http://{serverHost}:{serverPort}/debug/complete_all";
            using (var req = UnityWebRequest.PostWwwForm(url, ""))
            {
                req.timeout = 30;
                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning($"[QuestManager] 전체 완료 실패: {req.error}");
                    yield break;
                }

                DebugCompleteAllResponse resp = null;
                try
                {
                    resp = JsonUtility.FromJson<DebugCompleteAllResponse>(
                        req.downloadHandler.text);
                }
                catch { }

                int n = resp != null ? resp.completed : 0;
                Debug.Log($"[QuestManager] H키 — 진행 중 퀘스트 {n}개 완료");

                // 로컬 리스트에 있는 항목은 상태 갱신 + 반응 이벤트 (대화창 표시)
                if (resp != null && resp.results != null)
                {
                    foreach (var r in resp.results)
                    {
                        var entry = quests.Find(q => q.id == r.quest_id);
                        if (entry != null && entry.state != QuestState.Completed)
                        {
                            entry.state = QuestState.Completed;
                            OnQuestCompleted?.Invoke(entry, r.reaction ?? "");
                        }
                    }
                }
                OnQuestsChanged?.Invoke();
                // 현재 NPC 리스트 서버 동기화
                if (!string.IsNullOrEmpty(currentNpc))
                    FetchQuestsFor(currentNpc);
            }
        }

        /// <summary>현재 대화 NPC의 퀘스트 리스트를 서버에서 가져옴.</summary>
        public void FetchQuestsFor(string npc)
        {
            if (string.IsNullOrEmpty(npc)) return;
            currentNpc = npc;
            StartCoroutine(FetchCoroutine(npc));
        }

        IEnumerator FetchCoroutine(string npc)
        {
            string url = $"http://{serverHost}:{serverPort}/quests/{UnityWebRequest.EscapeURL(npc)}";
            using (var req = UnityWebRequest.Get(url))
            {
                req.timeout = 15;
                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning($"[QuestManager] 퀘스트 조회 실패: {req.error} ({url})");
                    yield break;
                }

                QuestListResponse resp;
                try
                {
                    resp = JsonUtility.FromJson<QuestListResponse>(req.downloadHandler.text);
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[QuestManager] 파싱 실패: {e.Message}");
                    yield break;
                }

                quests.Clear();
                if (resp != null && resp.quests != null)
                {
                    foreach (var q in resp.quests)
                    {
                        quests.Add(new QuestEntry
                        {
                            id = q.id,
                            title = q.title,
                            description = q.description,
                            reward = q.reward,
                            giver = resp.npc,
                            trustRequired = q.trust_required,
                            eligible = q.eligible,
                            state = ParseState(q.state),
                        });
                    }
                }
                OnQuestsChanged?.Invoke();
            }
        }

        static QuestState ParseState(string s)
        {
            switch ((s ?? "").ToLowerInvariant())
            {
                case "accepted": return QuestState.Accepted;
                case "completed": return QuestState.Completed;
                default: return QuestState.Available;   // available / offered(구버전)
            }
        }

        /// <summary>완료 처리 — 서버 trust +10. 진행 중(Accepted) quest만.</summary>
        public void CompleteQuest(QuestEntry entry)
        {
            if (entry == null || entry.state != QuestState.Accepted) return;
            StartCoroutine(CompleteCoroutine(entry));
        }

        IEnumerator CompleteCoroutine(QuestEntry entry)
        {
            string questIdParam = string.IsNullOrEmpty(entry.id)
                ? "" : $"?quest_id={UnityWebRequest.EscapeURL(entry.id)}";
            string url = $"http://{serverHost}:{serverPort}/quest_complete/" +
                         $"{UnityWebRequest.EscapeURL(entry.giver)}{questIdParam}";

            using (var req = UnityWebRequest.PostWwwForm(url, ""))
            {
                req.timeout = 30;
                yield return req.SendWebRequest();

                if (req.result == UnityWebRequest.Result.Success)
                {
                    entry.state = QuestState.Completed;
                    // NPC 완료 반응 대사 파싱 ("고맙다, 역시 너야")
                    string reaction = "";
                    try
                    {
                        var resp = JsonUtility.FromJson<QuestCompleteResponse>(
                            req.downloadHandler.text);
                        if (resp != null) reaction = resp.reaction ?? "";
                    }
                    catch { /* 반응 없이도 완료는 유효 */ }

                    Debug.Log($"[QuestManager] 퀘스트 완료: {entry.title} → {entry.giver} 친밀도 +10");
                    OnQuestsChanged?.Invoke();
                    OnQuestCompleted?.Invoke(entry, reaction);
                }
                else
                {
                    Debug.LogWarning($"[QuestManager] 완료 실패: {req.error} ({url})");
                }
            }
        }
    }
}
