using System;
using System.Threading.Tasks;
using NativeWebSocket;
using UnityEngine;

namespace NpcChat
{
    /// <summary>
    /// 단일 NPC와의 WebSocket 연결을 관리.
    /// 메시지 수신은 메인 스레드에서 콜백으로 디스패치.
    /// </summary>
    public class NpcChatClient
    {
        public string ServerHost = "127.0.0.1";
        public int ServerPort = 8000;
        public string Npc { get; private set; }

        public event Action OnReady;
        public event Action<ServerMessage> OnResponse;       // type=response
        public event Action<ServerMessage> OnTickEvents;     // type=tick_events
        public event Action<string> OnError;
        public event Action OnClosed;

        private WebSocket _ws;
        public bool IsOpen => _ws != null && _ws.State == WebSocketState.Open;

        public NpcChatClient(string npc) { Npc = NormalizeNpcName(npc); }

        /// <summary>
        /// 서버 등록 이름은 소문자 영문(elias/hermann/mathilda/finn/bernhardt).
        /// 한글이나 대소문자 혼합으로 들어와도 자동 변환.
        /// </summary>
        public static string NormalizeNpcName(string raw)
        {
            if (string.IsNullOrEmpty(raw)) return raw;
            string s = raw.Trim();
            // 한글 이름 매핑
            switch (s)
            {
                case "엘리아스": return "elias";
                case "헤르만":   return "hermann";
                case "마틸다":   return "mathilda";
                case "핀":       return "finn";
                case "베른하르트": return "bernhardt";
            }
            return s.ToLowerInvariant();
        }

        public async Task ConnectAsync()
        {
            if (_ws != null) await CloseAsync();

            string url = $"ws://{ServerHost}:{ServerPort}/ws/{Npc}";
            _ws = new WebSocket(url);

            _ws.OnOpen += () => Debug.Log($"[NpcChat] {Npc} 연결됨");
            _ws.OnError += (e) => { Debug.LogError($"[NpcChat] {Npc} 에러: {e}"); OnError?.Invoke(e); };
            _ws.OnClose += (_) => { Debug.Log($"[NpcChat] {Npc} 연결 종료"); OnClosed?.Invoke(); };
            _ws.OnMessage += HandleMessage;

            _ = _ws.Connect();
        }

        private void HandleMessage(byte[] bytes)
        {
            string json = System.Text.Encoding.UTF8.GetString(bytes);
            ServerMessage msg;
            try
            {
                msg = JsonUtility.FromJson<ServerMessage>(json);
            }
            catch (Exception e)
            {
                Debug.LogError($"[NpcChat] JSON 파싱 실패: {e.Message}\n원문: {json}");
                return;
            }

            switch (msg.type)
            {
                case "ready":
                    OnReady?.Invoke();
                    break;
                case "response":
                    OnResponse?.Invoke(msg);
                    break;
                case "tick_events":
                    OnTickEvents?.Invoke(msg);
                    break;
                case "reset_ok":
                    break;
                case "error":
                    OnError?.Invoke(msg.message);
                    break;
                default:
                    Debug.LogWarning($"[NpcChat] 알 수 없는 메시지 타입: {msg.type}");
                    break;
            }
        }

        public async Task SendChatAsync(string text)
        {
            if (!IsOpen)
            {
                OnError?.Invoke("연결이 열려있지 않습니다");
                return;
            }
            string json = JsonUtility.ToJson(new ChatRequest(text));
            await _ws.SendText(json);
        }

        public async Task SendTimeAdvanceAsync()
        {
            if (!IsOpen)
            {
                OnError?.Invoke("연결이 열려있지 않습니다");
                return;
            }
            string json = JsonUtility.ToJson(new TimeAdvanceRequest());
            await _ws.SendText(json);
        }

        public void DispatchQueue()
        {
            #if !UNITY_WEBGL || UNITY_EDITOR
            _ws?.DispatchMessageQueue();
            #endif
        }

        public async Task CloseAsync()
        {
            if (_ws == null) return;
            await _ws.Close();
            _ws = null;
        }
    }
}
