using System.Text;
using TMPro;
using UnityEngine;

namespace NpcChat
{
    /// <summary>
    /// 화면 좌상단 "Day N" 표시 + 시간 진행 안내.
    /// GameTimeController 이벤트 구독하여 자동 갱신.
    /// </summary>
    public class GameTimeHud : MonoBehaviour
    {
        [Header("표시 대상")]
        public TMP_Text dayLabel;       // "Day 1"
        public TMP_Text hintLabel;      // "N: 시간 진행"
        public TMP_Text statusLabel;    // "..." (진행 중)

        [Header("문구")]
        public string dayFormat = "Day {0}";
        public string hintText = "[N] 시간 진행";
        public string busyText = "마을 시간 흐르는 중...";
        public string idleText = "";

        void Start()
        {
            UpdateDay(0);
            if (hintLabel != null) hintLabel.text = hintText;
            if (statusLabel != null) statusLabel.text = idleText;

            if (GameTimeController.Instance != null)
            {
                GameTimeController.Instance.OnTickStarted += HandleStarted;
                GameTimeController.Instance.OnTickCompleted += HandleCompleted;
                GameTimeController.Instance.OnTickFailed += HandleFailed;
                UpdateDay(GameTimeController.Instance.currentDay);
            }
        }

        void OnDestroy()
        {
            if (GameTimeController.Instance != null)
            {
                GameTimeController.Instance.OnTickStarted -= HandleStarted;
                GameTimeController.Instance.OnTickCompleted -= HandleCompleted;
                GameTimeController.Instance.OnTickFailed -= HandleFailed;
            }
        }

        void HandleStarted()
        {
            if (statusLabel != null) statusLabel.text = busyText;
        }

        void HandleCompleted(ServerMessage msg)
        {
            UpdateDay(msg.day);
            if (statusLabel == null) return;

            var sb = new StringBuilder();
            int ev = msg.events != null ? msg.events.Length : 0;
            sb.Append($"<color=#bcaaa4>전파 {ev}건</color>");
            if (msg.turns != null && msg.turns.Length > 0
                && !string.IsNullOrEmpty(msg.npc_a) && !string.IsNullOrEmpty(msg.npc_b))
            {
                sb.Append($"  <color=#aaa>·</color>  ");
                sb.Append($"<color=#80cbc4>{msg.npc_a} ↔ {msg.npc_b}</color>");
            }
            statusLabel.text = sb.ToString();
            CancelInvoke(nameof(ClearStatus));
            Invoke(nameof(ClearStatus), 6f);
        }

        void HandleFailed(string err)
        {
            if (statusLabel != null) statusLabel.text = $"<color=#ef5350>에러: {err}</color>";
            CancelInvoke(nameof(ClearStatus));
            Invoke(nameof(ClearStatus), 5f);
        }

        void ClearStatus()
        {
            if (statusLabel != null) statusLabel.text = idleText;
        }

        void UpdateDay(int d)
        {
            if (dayLabel != null) dayLabel.text = string.Format(dayFormat, d);
        }
    }
}
