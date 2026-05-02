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
        public event Action<ChatResponse> OnResponse;
        public event Action<string> OnError;
        public event Action OnClosed;

        private WebSocket _ws;
        public bool IsOpen => _ws != null && _ws.State == WebSocketState.Open;

        public NpcChatClient(string npc) { Npc = npc; }

        public async Task ConnectAsync()
        {
            if (_ws != null) await CloseAsync();

            string url = $"ws://{ServerHost}:{ServerPort}/ws/{Npc}";
            _ws = new WebSocket(url);

            _ws.OnOpen += () => Debug.Log($"[NpcChat] {Npc} 연결됨");
            _ws.OnError += (e) => { Debug.LogError($"[NpcChat] {Npc} 에러: {e}"); OnError?.Invoke(e); };
            _ws.OnClose += (_) => { Debug.Log($"[NpcChat] {Npc} 연결 종료"); OnClosed?.Invoke(); };
            _ws.OnMessage += HandleMessage;

            // 비동기 연결 — await하면 OnOpen 콜백이 먼저 fire됨
            _ = _ws.Connect();
        }

        private void HandleMessage(byte[] bytes)
        {
            string json = System.Text.Encoding.UTF8.GetString(bytes);
            ChatResponse msg;
            try
            {
                msg = JsonUtility.FromJson<ChatResponse>(json);
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

        /// <summary>메인 스레드 Update에서 매 프레임 호출 필수 (non-WebGL).</summary>
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
