using System;

namespace NpcChat
{
    [Serializable]
    public class ChatRequest
    {
        public string type = "chat";
        public string text;

        public ChatRequest(string text)
        {
            this.text = text;
        }
    }

    [Serializable]
    public class TimeAdvanceRequest
    {
        public string type = "time_advance";
    }

    /// <summary>
    /// 서버 → 클라이언트 메시지 (모든 type 통합).
    /// JsonUtility 한계로 nested array는 한 단계만.
    /// </summary>
    [Serializable]
    public class ServerMessage
    {
        public string type;     // ready | response | reset_ok | tick_events | error

        // chat response
        public string npc;
        public string text;
        public int latency_ms;
        public Memory[] memories_used;

        // tick events
        public int day;
        public TickEvent[] events;
        public MemoryCount[] memory_counts;  // (사용 안 함, 아래 dict는 JsonUtility 미지원)

        // error
        public string message;
    }

    [Serializable]
    public class Memory
    {
        public string text;
        public int importance;
        public string source;   // seed | observation | dialogue | propagation
    }

    [Serializable]
    public class TickEvent
    {
        public int day;
        public string from;
        public string to;
        public string original;
        public string transformed;
        public int importance_before;
        public int importance_after;
    }

    [Serializable]
    public class MemoryCount
    {
        public string npc;
        public int count;
    }
}
