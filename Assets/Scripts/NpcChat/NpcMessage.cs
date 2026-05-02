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
    public class ChatResponse
    {
        public string type;     // "ready" | "response" | "error"
        public string npc;
        public string text;
        public string message;  // 에러 시
        public int latency_ms;
        public Memory[] memories_used;
    }

    [Serializable]
    public class Memory
    {
        public string text;
        public int importance;
        public string source;   // "seed" | "observation" | "dialogue" | "propagation"
    }
}
