#ifndef FRAME_RING_BUFFER_H
#define FRAME_RING_BUFFER_H

#include <functional>
#include <future>

// Shared "2 active + 1 preloading" frame cache used by WindInterpolator
// (CARRA1) and CurrentInterpolator (GLO12): 3 physical slots, of which
// exactly 2 are "active" (the pair currently being interpolated between)
// and at most 1 asynchronous background load is ever in flight, into the
// third/spare slot. Owns only the frame-index bookkeeping and the preload's
// synchronization -- callers own the actual data buffers and supply a
// load_func(frameIdx, slotIdx) that fills them.
//
// Invariants this enforces (previously each caller reimplemented this with
// different, incompatible bugs -- see history):
//  - Never more than one std::async load in flight at a time.
//  - Update() never blocks on a preload for a frame that isn't one of the
//    two needed frames this call (i.e. never waits on the "3rd", not-yet-
//    active frame).
//  - Update() DOES block if a needed frame isn't resident yet -- either
//    waiting for it specifically (if it's the in-flight preload's target)
//    or loading it synchronously right now (if no preload covers it).
class FrameRingBuffer
{
public:
    static constexpr int NUM_SLOTS = 3;

    // Ensures physical slots exist for logical frames needed_f0/needed_f1
    // (blocking only as described above), then -- if no preload is
    // currently in flight and preload_candidate >= 0 and isn't already
    // resident -- kicks off an async load of it into the spare slot.
    // load_func is used for both synchronous loads (called directly) and
    // the async preload's body (called on a background thread).
    // Returns true if the active frame pair actually changed.
    bool Update(int needed_f0, int needed_f1, int preload_candidate,
                const std::function<void(int, int)>& load_func)
    {
        const int current_f0 = (active_slots_[0] == -1) ? -1 : slot_frames_[active_slots_[0]];
        const int current_f1 = (active_slots_[1] == -1) ? -1 : slot_frames_[active_slots_[1]];
        const bool changed = (current_f0 != needed_f0 || current_f1 != needed_f1);

        if (changed) {
            const int s0 = ResolveFrame(needed_f0, -1, load_func);
            const int s1 = ResolveFrame(needed_f1, s0, load_func);
            active_slots_[0] = s0;
            active_slots_[1] = s1;
        }

        if (preload_enabled_ && preload_target_frame_ == -1 && preload_candidate >= 0 &&
            FindSlot(preload_candidate) == -1) {
            int spare = -1;
            for (int i = 0; i < NUM_SLOTS; ++i) {
                if (i != active_slots_[0] && i != active_slots_[1]) { spare = i; break; }
            }
            if (spare != -1) {
                slot_frames_[spare] = preload_candidate;
                preload_target_frame_ = preload_candidate;
                preload_target_slot_ = spare;
                preload_future_ = std::async(std::launch::async, [load_func, preload_candidate, spare]() {
                    load_func(preload_candidate, spare);
                });
            }
        }

        return changed;
    }

    int ActiveSlot(int logicalFrame) const { return active_slots_[logicalFrame]; }
    bool IsReady() const { return active_slots_[0] != -1 && active_slots_[1] != -1; }

    // Disables launching new background preloads (default: enabled). Useful
    // for trackbar-driven callers (preparer, visualizer) that jump around
    // non-sequentially/in both directions, where preloading "the next
    // frame" is usually wasted work. Doesn't affect an already-in-flight
    // preload; Update() still blocks on it if the frame it's needed for.
    void SetPreloadEnabled(bool enabled) { preload_enabled_ = enabled; }

    // Blocks until any in-flight preload completes, then clears all
    // bookkeeping (as if nothing were ever loaded). Call before swapping
    // out / destroying the underlying data buffers -- otherwise a
    // background load could still be writing into them.
    void Reset()
    {
        if (preload_future_.valid()) preload_future_.wait();
        for (int i = 0; i < NUM_SLOTS; ++i) slot_frames_[i] = -1;
        active_slots_[0] = active_slots_[1] = -1;
        preload_target_frame_ = -1;
        preload_target_slot_ = -1;
    }

private:
    int slot_frames_[NUM_SLOTS] = {-1, -1, -1};
    int active_slots_[2] = {-1, -1};

    std::future<void> preload_future_;
    int preload_target_frame_ = -1;
    int preload_target_slot_ = -1;
    bool preload_enabled_ = true;

    int FindSlot(int frame) const
    {
        for (int i = 0; i < NUM_SLOTS; ++i) if (slot_frames_[i] == frame) return i;
        return -1;
    }

    // Resolves one needed frame to a slot, blocking only if necessary.
    // avoid_slot: the slot just assigned to the sibling frame this same
    // Update() call (-1 if none yet) -- never picked as an eviction target.
    int ResolveFrame(int frame, int avoid_slot, const std::function<void(int, int)>& load_func)
    {
        int slot = FindSlot(frame);
        if (slot != -1) return slot;

        if (preload_target_frame_ == frame) {
            preload_future_.wait();
            slot = preload_target_slot_;
            preload_target_frame_ = -1;
            preload_target_slot_ = -1;
            return slot;
        }

        // Not resident anywhere and not the in-flight preload -- load it
        // synchronously now, into a slot that's neither this round's
        // sibling nor (if some other preload is still running) its target.
        for (int i = 0; i < NUM_SLOTS; ++i) {
            if (i == avoid_slot) continue;
            if (preload_target_frame_ != -1 && i == preload_target_slot_) continue;
            slot = i;
            break;
        }
        load_func(frame, slot);
        slot_frames_[slot] = frame;
        return slot;
    }
};

#endif // FRAME_RING_BUFFER_H
