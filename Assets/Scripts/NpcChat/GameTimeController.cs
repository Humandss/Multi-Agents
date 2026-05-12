using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

namespace NpcChat
{
    /// <summary>
    /// 게임 월드 항상 활성. T키로 시간 진행 (HTTP POST /tick).
    /// WebSocket 의존성 없음 — dialogue 안/밖 어디서나 작동.
    ///
    /// 결과:
    ///   - OnTickStarted: 진행 시작
    ///   - OnTickCompleted(ServerMessage): propagation events + npc-npc conversation
    ///   - OnTickFailed(string): 에러 메시지
    /// </summary>
    public class GameTimeController : MonoBehaviour
    {
        public static GameTimeController Instance { get; private set; }

        [Header("서버")]
        public string serverHost = "127.0.0.1";
        public int serverPort = 8000;
        public int numTurns = 2;
        public bool npcConversation = true;

        [Header("입력")]
        public KeyCode advanceKey = KeyCode.N;  // T는 마을 이름이 잡고 있음 → N (Next)
        public bool blockWhenInputFocused = true;

        [Header("상태")]
        public int currentDay = 0;
        public bool busy { get; private set; }

        public event Action OnTickStarted;
        public event Action<ServerMessage> OnTickCompleted;
        public event Action<string> OnTickFailed;

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
            if (busy) return;
            if (!Input.GetKeyDown(advanceKey)) return;
            if (blockWhenInputFocused && IsInputFieldFocused()) return;

            AdvanceTime();
        }

        static bool IsInputFieldFocused()
        {
            var sel = UnityEngine.EventSystems.EventSystem.current?.currentSelectedGameObject;
            if (sel == null) return false;
            if (sel.GetComponent<TMPro.TMP_InputField>() != null) return true;
            if (sel.GetComponent<UnityEngine.UI.InputField>() != null) return true;
            return false;
        }

        public void AdvanceTime()
        {
            if (busy) return;
            StartCoroutine(TickCoroutine());
        }

        IEnumerator TickCoroutine()
        {
            busy = true;
            OnTickStarted?.Invoke();

            string url = $"http://{serverHost}:{serverPort}/tick?" +
                         $"npc_conversation={(npcConversation ? "true" : "false")}" +
                         $"&num_turns={numTurns}";

            using (var req = UnityWebRequest.PostWwwForm(url, ""))
            {
                req.timeout = 600;  // tick은 1-2분 걸릴 수 있음
                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    busy = false;
                    OnTickFailed?.Invoke($"{req.error} (url: {url})");
                    yield break;
                }

                string json = req.downloadHandler.text;
                ServerMessage msg;
                try
                {
                    msg = JsonUtility.FromJson<ServerMessage>(json);
                }
                catch (Exception e)
                {
                    busy = false;
                    OnTickFailed?.Invoke($"JSON 파싱 실패: {e.Message}");
                    yield break;
                }

                currentDay = msg.day > 0 ? msg.day : currentDay + 1;
                busy = false;
                OnTickCompleted?.Invoke(msg);
            }
        }
    }
}
