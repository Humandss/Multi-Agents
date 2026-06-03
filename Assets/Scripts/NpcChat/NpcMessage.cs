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
        public Quest quest;     // null 또는 비어있으면 quest 없음

        // trust/friendship
        public int trust;          // 0-100 (default 30)
        public string trust_label; // 낯선 사람 / 지인 / 친구 / 절친
        public int trust_delta;    // 이번 turn 변화량 (+1 etc.)

        // tick events
        public int day;
        public TickEvent[] events;
        public MemoryCount[] memory_counts;  // (사용 안 함, 아래 dict는 JsonUtility 미지원)

        // npc-npc conversation (Phase 2)
        public string npc_a;
        public string npc_b;
        public string topic;
        public ConversationTurn[] turns;
        public bool memory_saved;

        // error
        public string message;
    }

    [Serializable]
    public class ConversationTurn
    {
        public string speaker;
        public string speaker_ko;
        public string text;
    }

    [Serializable]
    public class Memory
    {
        public string text;
        public int importance;
        public string source;   // seed | observation | dialogue | propagation
    }

    /// <summary>
    /// NPC가 생성한 quest. JsonUtility 한계로 null 체크는 title 비어있는지로 판단.
    /// </summary>
    [Serializable]
    public class Quest
    {
        public string title;
        public string description;
        public string reward;
        public string giver;

        public bool IsValid => !string.IsNullOrEmpty(title);
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
