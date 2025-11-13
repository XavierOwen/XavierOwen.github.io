---
layout: single
title: "All Tags"
permalink: /tags/
---

{% comment %}
  <!-- Get all tags from all collections and posts -->
{% endcomment %}
{% assign all_tags = "" | split: "" %}

{% comment %}<!-- Get tags from posts -->{% endcomment %}
{% for post in site.posts %}
  {% if post.tags %}
    {% assign all_tags = all_tags | concat: post.tags %}
  {% endif %}
{% endfor %}

{% comment %}<!-- Get tags from notes -->{% endcomment %}
{% for note in site.notes %}
  {% if note.tags %}
    {% assign all_tags = all_tags | concat: note.tags %}
  {% endif %}
{% endfor %}

{% comment %}<!-- Get tags from spirits -->{% endcomment %}
{% for spirit in site.spirits %}
  {% if spirit.tags %}
    {% assign all_tags = all_tags | concat: spirit.tags %}
  {% endif %}
{% endfor %}

{% comment %}<!-- Get tags from projects -->{% endcomment %}
{% for project in site.projects %}
  {% if project.tags %}
    {% assign all_tags = all_tags | concat: project.tags %}
  {% endif %}
{% endfor %}

{% assign all_tags = all_tags | uniq | sort %}

<p><em>Found {{ all_tags.size }} unique tags</em></p>

<div class="tags-grid">
  {% for tag in all_tags %}
    {% comment %}
      <!-- Count posts for each tag -->
    {% endcomment %}
    {% assign tag_posts = site.posts | where: "tags", tag %}
    {% assign tag_notes = site.notes | where: "tags", tag %}
    {% assign tag_spirits = site.spirits | where: "tags", tag %}
    {% assign tag_projects = site.projects | where: "tags", tag %}
    {% assign tag_count = tag_posts.size | plus: tag_notes.size | plus: tag_spirits.size | plus: tag_projects.size %}

    <a href="/tags/{{ tag | slugify }}/" class="tag-cloud-item">
      <span class="tag-name">{{ tag }}</span>
      <span class="tag-count">{{ tag_count }}</span>
    </a>
  {% endfor %}
</div>

<style>
.tags-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1em;
  margin-top: 1em;
}

.tag-cloud-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 60px;
  padding: 0.8em 1.2em;
  background-color: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 6px;
  text-decoration: none;
  color: #333;
  font-weight: 500;
  transition: all 0.2s ease-in-out;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.tag-cloud-item:hover {
  background-color: #0366d6;
  color: white;
  border-color: #0366d6;
  text-decoration: none;
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(3,102,214,0.2);
}

.tag-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-right: 0.5em;
}

.tag-count {
  font-size: 0.8em;
  opacity: 0.7;
  font-weight: 600;
  background-color: rgba(0,0,0,0.1);
  padding: 0.2em 0.5em;
  border-radius: 12px;
  min-width: 30px;
  text-align: center;
}

.tag-cloud-item:hover .tag-count {
  opacity: 1;
  background-color: rgba(255,255,255,0.2);
}

@media (max-width: 768px) {
  .tags-grid {
    grid-template-columns: 1fr;
  }
  
  .tag-cloud-item {
    height: 50px;
    font-size: 0.85em;
  }
}

@media (min-width: 1025px) {
  .tags-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}
</style>