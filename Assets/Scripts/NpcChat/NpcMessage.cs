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

    [Serializable]
    public class QuestProposeRequest
    {
        public string type = "quest_propose";
        public string quest_id;

        public QuestProposeRequest(string questId) { quest_id = questId; }
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

        // quest 흐름 (propose → 수락/거절)
        public string quest_stage; // proposed | accepted | declined | unclear
        public string quest_id;

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
        public string id;       // quest pool id (예: hermann_meteor_ore) — 완료 API에 사용
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

    /// <summary>GET /quests/{npc} 응답 — 퀘스트 리스트 패널용.</summary>
    [Serializable]
    public class QuestInfo
    {
        public string id;
        public string title;
        public string description;
        public string reward;
        public int trust_required;
        public string state;     // available | accepted | completed
        public bool eligible;    // 지금 시작 가능한가 (trust 충족 + available)
    }

    [Serializable]
    public class QuestListResponse
    {
        public string npc;
        public int trust;
        public QuestInfo[] quests;
    }

    /// <summary>POST /quest_complete 응답 — NPC 완료 반응 대사 포함.</summary>
    [Serializable]
    public class QuestCompleteResponse
    {
        public string npc;
        public string quest_id;
        public string title;
        public string reaction;    // "고맙다, 역시 너야" — 대화창 표시용
        public int trust;
        public string trust_label;
        public int trust_delta;
    }

    /// <summary>POST /debug/complete_all 응답 (H키 — 진행 중 퀘스트 전체 완료).</summary>
    [Serializable]
    public class DebugCompleteAllResponse
    {
        public int completed;
        public QuestCompleteResponse[] results;
    }

    /// <summary>POST /debug/toggle_transform 응답 (T키 — 내용 왜곡 ON/OFF).</summary>
    [Serializable]
    public class ToggleTransformResponse
    {
        public bool content_distortion;
    }
}
