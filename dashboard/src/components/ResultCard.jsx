import React from 'react';
import { Download, Share2, Instagram, Youtube, Video } from 'lucide-react';

export default function ResultCard({ clip, index }) {
    // clip contains: start, end, video_description_for_tiktok, etc.
    // We also expect a videoUrl or similar from the API response

    return (
        <div className="glass-panel overflow-hidden animate-[fadeIn_0.5s_ease-out]" style={{ animationDelay: `${index * 0.1}s` }}>
            <div className="grid md:grid-cols-2 gap-0">
                <div className="bg-black/50 aspect-[9/16] relative group">
                    <video
                        src={clip.video_url}
                        controls
                        className="w-full h-full object-contain"
                        loop
                    />
                    <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <a
                            href={clip.video_url}
                            download
                            className="p-2 bg-black/60 backdrop-blur rounded-lg text-white hover:bg-black/80 inline-flex"
                        >
                            <Download size={16} />
                        </a>
                    </div>
                </div>

                <div className="p-6 flex flex-col h-full">
                    <div className="mb-4">
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-xs font-bold px-2 py-1 bg-primary/20 text-primary rounded-md uppercase tracking-wider">
                                Clip {index + 1}
                            </span>
                            <span className="text-xs text-zinc-500 font-mono">
                                {clip.start}s - {clip.end}s
                            </span>
                            <span className="ml-auto text-xs font-medium text-red-400 flex items-center gap-1">
                                <Youtube size={12} /> YouTube Short Title
                            </span>
                        </div>
                        <h3 className="text-xl font-bold leading-tight mb-4 text-white">
                            {clip.video_title_for_youtube_short || "Viral Clip"}
                        </h3>
                    </div>

                    <div className="space-y-4 flex-1">
                        <div className="space-y-1">
                            <div className="flex items-center gap-2 text-xs font-medium text-pink-400">
                                <Instagram size={14} /> Instagram Description
                            </div>
                            <div className="p-3 bg-white/5 rounded-lg text-sm text-zinc-300 leading-relaxed border border-white/5 line-clamp-3 hover:line-clamp-none transition-all">
                                {clip.video_description_for_instagram}
                            </div>
                        </div>

                        <div className="space-y-1">
                            <div className="flex items-center gap-2 text-xs font-medium text-cyan-400">
                                <Video size={14} /> TikTok Description
                            </div>
                            <div className="p-3 bg-white/5 rounded-lg text-sm text-zinc-300 leading-relaxed border border-white/5 line-clamp-3 hover:line-clamp-none transition-all">
                                {clip.video_description_for_tiktok || clip.video_description_for_instagram}
                            </div>
                        </div>
                    </div>

                    <div className="mt-6 pt-4 border-t border-white/10 flex gap-2">
                        <button className="flex-1 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2">
                            <Share2 size={16} /> Share
                        </button>
                        <a
                            href={clip.video_url}
                            download
                            className="flex-1 py-2 bg-primary/10 hover:bg-primary/20 text-primary rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
                        >
                            <Download size={16} /> Download
                        </a>
                    </div>
                </div>
            </div>
        </div>
    );
}
